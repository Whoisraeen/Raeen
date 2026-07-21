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
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// The process already owns the maximum number of emulator AudioOut ports.
const SCE_AUDIO_OUT_ERROR_PORT_FULL: u64 = 0x8026_0004;
/// Upper bound on how long one `Output` blocks — a wild grain/frequency
/// can't wedge the audio thread for more than this.
const OUTPUT_MAX_SLEEP: Duration = Duration::from_millis(100);
/// Sanity cap on one `Output` buffer read from the guest. Grain is typically
/// 256–2048 samples; 8ch × f32 × a generous grain stays well under this.
const MAX_OUTPUT_BYTES: usize = 1 << 20;

/// A port's PCM layout, decoded from `sceAudioOutOpen`'s `param` (which the
/// kernel's `(grain, freq)` record does not keep). Read by `Output` to
/// interpret the guest buffer.
#[derive(Clone, Copy)]
struct PortFormat {
    channels: u32,
    is_float: bool,
}

impl Default for PortFormat {
    fn default() -> Self {
        // Stereo S16 — the common case, and a safe fallback for an unknown port.
        Self {
            channels: 2,
            is_float: false,
        }
    }
}

/// Decode `SceAudioOutParamFormat` (the low bits of `sceAudioOutOpen`'s `param`)
/// into channel count + sample type. Unknown values fall back to stereo S16.
fn decode_format(param: u64) -> PortFormat {
    match param & 0xF {
        0 => PortFormat {
            channels: 1,
            is_float: false,
        }, // S16_MONO
        1 => PortFormat {
            channels: 2,
            is_float: false,
        }, // S16_STEREO
        2 | 6 => PortFormat {
            channels: 8,
            is_float: false,
        }, // S16_8CH[_STD]
        3 => PortFormat {
            channels: 1,
            is_float: true,
        }, // FLOAT_MONO
        4 => PortFormat {
            channels: 2,
            is_float: true,
        }, // FLOAT_STEREO
        5 | 7 => PortFormat {
            channels: 8,
            is_float: true,
        }, // FLOAT_8CH[_STD]
        _ => PortFormat::default(),
    }
}

/// Per-handle PCM layout table (the kernel port records only grain+freq).
/// Process-global; `Open` overwrites and `Close` removes, and handles are
/// reused per-process, so a stale entry never leaks into a new port.
fn port_formats() -> &'static Mutex<HashMap<u32, PortFormat>> {
    static FORMATS: OnceLock<Mutex<HashMap<u32, PortFormat>>> = OnceLock::new();
    FORMATS.get_or_init(|| Mutex::new(HashMap::new()))
}

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
    // `param` (arg 6) carries the PCM format; default to stereo S16 when absent.
    let format = decode_format(args.get(5).copied().unwrap_or(1));
    port_formats()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(handle, format);
    debug!(
        "sceAudioOutOpen(grain={grain}, freq={freq}, ch={}, float={}) -> handle {handle}",
        format.channels, format.is_float
    );
    handle as u64
}

/// `sceAudioOutOutput(handle, ptr)`: read the guest PCM buffer, hand it to the
/// host mixer as stereo f32, sleep ~one buffer period (grain ÷ freq, capped) so
/// the audio thread paces to real time instead of spinning, and return the
/// sample count. The pacing sleep also matches the host consumption rate, so
/// the ring stays shallow.
fn hle_output(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let ptr = args.get(1).copied().unwrap_or(0);
    let (grain, freq) = ctx.kernel.audio_out_port(handle).unwrap_or((256, 48_000));
    let format = port_formats()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&handle)
        .copied()
        .unwrap_or_default();

    if ptr != 0 {
        let bytes_per_sample = if format.is_float { 4 } else { 2 };
        let byte_len = grain as usize * format.channels as usize * bytes_per_sample;
        if byte_len > 0 && byte_len <= MAX_OUTPUT_BYTES {
            let mut buf = vec![0u8; byte_len];
            if ctx.mem.read(ptr, &mut buf) {
                let stereo = decode_to_stereo(&buf, format);
                xps5x_audio::output::submit(freq, &stereo);
            }
        }
    }

    if freq > 0 {
        let period = Duration::from_secs_f64(grain as f64 / freq as f64).min(OUTPUT_MAX_SLEEP);
        std::thread::sleep(period);
    }
    grain as u64
}

/// Decode an interleaved guest PCM buffer to interleaved-stereo f32: the front
/// L/R of a multichannel stream, or mono duplicated to both channels.
fn decode_to_stereo(buf: &[u8], format: PortFormat) -> Vec<f32> {
    let channels = format.channels.max(1) as usize;
    let bps = if format.is_float { 4 } else { 2 };
    let frame_bytes = channels * bps;
    if frame_bytes == 0 {
        return Vec::new();
    }
    let frames = buf.len() / frame_bytes;
    let mut out = Vec::with_capacity(frames * 2);
    let sample = |base: usize, ch: usize| -> f32 {
        let o = base + ch * bps;
        if format.is_float {
            f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
        } else {
            f32::from(i16::from_le_bytes([buf[o], buf[o + 1]])) / 32768.0
        }
    };
    for f in 0..frames {
        let base = f * frame_bytes;
        let (l, r) = if channels == 1 {
            let m = sample(base, 0);
            (m, m)
        } else {
            (sample(base, 0), sample(base, 1))
        };
        out.push(l);
        out.push(r);
    }
    out
}

/// `sceAudioOutClose(handle)`: drop the port's recorded pacing + format state.
fn hle_close(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    debug!("sceAudioOutClose(handle={handle})");
    port_formats()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&handle);
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

    #[test]
    fn decode_to_stereo_handles_s16_mono_and_multichannel() {
        let s16 = |v: i16| v.to_le_bytes();

        // Stereo S16: one frame -> one interleaved [L, R] pair.
        let mut stereo = Vec::new();
        stereo.extend(s16(16_384)); // L ≈ +0.5
        stereo.extend(s16(-16_384)); // R ≈ -0.5
        let out = decode_to_stereo(
            &stereo,
            PortFormat {
                channels: 2,
                is_float: false,
            },
        );
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 0.01 && (out[1] + 0.5).abs() < 0.01);

        // Mono S16: duplicated to both channels.
        let out = decode_to_stereo(
            &s16(32_767),
            PortFormat {
                channels: 1,
                is_float: false,
            },
        );
        assert_eq!(out.len(), 2);
        assert!((out[0] - out[1]).abs() < 1e-6 && out[0] > 0.99);

        // 8ch float: only the front two channels are kept.
        let mut ch8 = Vec::new();
        for i in 0..8 {
            ch8.extend((i as f32 * 0.1).to_le_bytes());
        }
        let out = decode_to_stereo(
            &ch8,
            PortFormat {
                channels: 8,
                is_float: true,
            },
        );
        assert_eq!(out.len(), 2);
        assert!(out[0].abs() < 1e-6 && (out[1] - 0.1).abs() < 1e-6);
    }
}
