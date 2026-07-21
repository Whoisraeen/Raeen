//! Host audio output (WASAPI / PulseAudio / CoreAudio via cpal).
//!
//! The guest's `sceAudioOutOutput` (HLE `libSceAudioOut`) decodes its PCM
//! buffer to interleaved-stereo f32 and calls [`submit`]; a cpal output stream
//! on a dedicated thread drains a shared ring buffer to the speakers, applying
//! the user's master volume / enable from Settings ▸ Audio ([`set_volume`] /
//! [`set_enabled`]).
//!
//! Sample-rate matching: the stream opens at the guest's 48 kHz when the device
//! supports it (no resampling needed), otherwise at the device default, and
//! [`submit`] linearly resamples the guest audio to that rate so playback is
//! pitch-correct on 44.1 kHz devices. A missing output device, or one that won't
//! do f32, is never fatal — audio just stays silent.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tracing::{info, warn};

/// Interleaved-stereo f32 ring the HLE fills and the cpal callback drains.
type Ring = Arc<Mutex<VecDeque<f32>>>;

static RING: OnceLock<Ring> = OnceLock::new();
/// Master volume as `f32` bits (`Relaxed` is fine — a one-frame-late volume is
/// inaudible). `0` bits (== `0.0`) until [`init`] or [`set_volume`] sets it.
static VOLUME_BITS: AtomicU32 = AtomicU32::new(0);
static ENABLED: AtomicBool = AtomicBool::new(true);
/// The host device's actual output sample rate, set once when the stream is
/// built (`0` until then). Guest audio (usually 48 kHz) is resampled to this in
/// [`submit`] so playback is pitch-correct on 44.1 kHz devices.
static DEVICE_RATE: AtomicU32 = AtomicU32::new(0);

/// ~250 ms of stereo at 48 kHz. Submissions beyond this drop the oldest
/// samples, so a slow/stalled consumer can never grow memory without bound.
const MAX_RING_SAMPLES: usize = 48_000 * 2 / 4;

/// Streaming linear-resampler state (guest rate → device rate), carried across
/// [`submit`] calls so the read phase is continuous and buffer boundaries stay
/// click-free. `hist` is the previous buffer's last frame (virtual index −1).
#[derive(Default)]
struct Resampler {
    src_rate: u32,
    pos: f64,
    hist: [f32; 2],
}

fn resampler() -> &'static Mutex<Resampler> {
    static R: OnceLock<Mutex<Resampler>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Resampler::default()))
}

/// Set the master volume, `0.0..=1.0` (clamped). Settings ▸ Audio ▸ Master
/// Volume — takes effect immediately.
pub fn set_volume(volume: f32) {
    VOLUME_BITS.store(volume.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

/// Enable or mute output. Disabled drops incoming samples and silences the
/// stream. Settings ▸ Audio ▸ Audio Enabled — takes effect immediately.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

fn gain() -> f32 {
    if ENABLED.load(Ordering::Relaxed) {
        f32::from_bits(VOLUME_BITS.load(Ordering::Relaxed))
    } else {
        0.0
    }
}

/// Bring up the host output stream once. Idempotent, and a no-op when no output
/// device is available — audio simply stays silent (never a reason to fail to
/// boot). Call once at startup, after [`set_volume`] / [`set_enabled`].
pub fn init() {
    if RING.get().is_some() {
        return;
    }
    if VOLUME_BITS.load(Ordering::Relaxed) == 0 {
        // No volume set before init — default to full.
        VOLUME_BITS.store(1.0f32.to_bits(), Ordering::Relaxed);
    }
    let ring: Ring = Arc::new(Mutex::new(VecDeque::new()));
    let _ = RING.set(ring.clone());

    // `cpal::Stream` is `!Send` on some backends and stops when dropped, so
    // build it on a dedicated thread and park to keep it alive for the process
    // lifetime.
    let spawned = std::thread::Builder::new()
        .name("xps5x-audio".to_owned())
        .spawn(move || match build_stream(ring) {
            Ok(stream) => {
                if let Err(e) = stream.play() {
                    warn!("host audio: stream play failed ({e}); continuing silent");
                    return;
                }
                loop {
                    std::thread::park();
                }
            }
            Err(e) => warn!("host audio unavailable ({e}); continuing silent"),
        });
    if let Err(e) = spawned {
        warn!("host audio: could not start audio thread ({e})");
    }
}

fn err_cb(e: cpal::StreamError) {
    warn!("host audio: stream error ({e})");
}

fn build_stream(ring: Ring) -> anyhow::Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device"))?;

    // Prefer 48 kHz stereo f32 (the PS5 rate) so the guest samples need no
    // resampling.
    let wanted = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(48_000),
        buffer_size: cpal::BufferSize::Default,
    };
    let ring_wanted = ring.clone();
    match device.build_output_stream(
        &wanted,
        move |out: &mut [f32], _: &_| fill(out, 2, &ring_wanted),
        err_cb,
        None,
    ) {
        Ok(stream) => {
            DEVICE_RATE.store(48_000, Ordering::Relaxed);
            info!("host audio output ready (48000 Hz, stereo)");
            Ok(stream)
        }
        Err(_) => {
            // The device won't do exactly 48 kHz stereo f32 — use its default.
            let supported = device.default_output_config()?;
            let channels = supported.channels() as usize;
            let rate = supported.sample_rate().0;
            if supported.sample_format() != cpal::SampleFormat::F32 {
                return Err(anyhow::anyhow!(
                    "device default sample format is {:?}, not F32",
                    supported.sample_format()
                ));
            }
            let config: cpal::StreamConfig = supported.config();
            DEVICE_RATE.store(rate, Ordering::Relaxed);
            info!(
                "host audio output ready ({rate} Hz, {channels}ch; guest 48000 Hz resampled to match)"
            );
            let stream = device.build_output_stream(
                &config,
                move |out: &mut [f32], _: &_| fill(out, channels, &ring),
                err_cb,
                None,
            )?;
            Ok(stream)
        }
    }
}

/// cpal callback: drain the stereo ring into the device's frames, placing L/R
/// on the first two channels (others silent) and applying master gain. An empty
/// ring outputs silence (an underrun, not a glitch).
fn fill(out: &mut [f32], out_channels: usize, ring: &Ring) {
    let g = gain();
    let mut ring = ring
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for frame in out.chunks_mut(out_channels.max(1)) {
        let l = ring.pop_front().unwrap_or(0.0) * g;
        let r = ring.pop_front().unwrap_or(0.0) * g;
        for (i, s) in frame.iter_mut().enumerate() {
            *s = match i {
                0 => l,
                1 => r,
                _ => 0.0,
            };
        }
    }
}

/// Submit interleaved-stereo f32 samples from the guest at `src_rate` Hz
/// (already downmixed by the HLE). Resampled to the device's rate so playback
/// is pitch-correct (e.g. guest 48 kHz on a 44.1 kHz device), then appended to
/// the ring — dropping the oldest if the consumer is behind so memory stays
/// bounded. A no-op until [`init`], or when muted.
pub fn submit(src_rate: u32, samples: &[f32]) {
    let Some(ring) = RING.get() else {
        return;
    };
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let dst_rate = DEVICE_RATE.load(Ordering::Relaxed);

    // Resample guest→device before touching the ring, so the cpal callback is
    // never blocked on it. Matching (or unknown) rates pass straight through.
    let owned;
    let frames: &[f32] = if src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        samples
    } else {
        let mut state = resampler()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        owned = resample_stereo(&mut state, src_rate, dst_rate, samples);
        &owned
    };

    let mut ring = ring
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ring.extend(frames.iter().copied());
    if ring.len() > MAX_RING_SAMPLES {
        let excess = ring.len() - MAX_RING_SAMPLES;
        ring.drain(0..excess);
    }
}

/// Streaming linear resampler (interleaved stereo): convert `input` from
/// `src_rate` to `dst_rate`, carrying the fractional read position and one frame
/// of history in `state` so successive buffers join seamlessly (no boundary
/// clicks). Linear interpolation is ample for the small 48k↔44.1k ratio here.
fn resample_stereo(state: &mut Resampler, src_rate: u32, dst_rate: u32, input: &[f32]) -> Vec<f32> {
    let n = input.len() / 2;
    if n == 0 {
        return Vec::new();
    }
    // A change of source rate (a new port config) restarts the phase.
    if state.src_rate != src_rate {
        state.src_rate = src_rate;
        state.pos = 0.0;
        state.hist = [0.0, 0.0];
    }
    let ratio = f64::from(src_rate) / f64::from(dst_rate); // input frames per output frame
    let hist = state.hist;
    // Virtual input frame `i`: −1 → the carried `hist`, 0..n-1 → `input`.
    let frame = |i: isize| -> (f32, f32) {
        if i < 0 {
            (hist[0], hist[1])
        } else {
            let i = (i as usize).min(n - 1);
            (input[i * 2], input[i * 2 + 1])
        }
    };
    let mut out = Vec::with_capacity(((n as f64 / ratio) as usize + 2) * 2);
    let mut pos = state.pos;
    // Stop before `pos` would need the next buffer's first frame (index n); that
    // output is produced next call, where this buffer's last frame is `hist`.
    let limit = (n - 1) as f64;
    while pos < limit {
        let i0 = pos.floor() as isize;
        let frac = (pos - i0 as f64) as f32;
        let (l0, r0) = frame(i0);
        let (l1, r1) = frame(i0 + 1);
        out.push(l0 + (l1 - l0) * frac);
        out.push(r0 + (r1 - r0) * frac);
        pos += ratio;
    }
    // Carry this buffer's last frame as index −1 for the next call, and shift
    // the read position into the next buffer's frame numbering.
    state.hist = [input[(n - 1) * 2], input[(n - 1) * 2 + 1]];
    state.pos = pos - n as f64;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_before_init_is_a_silent_no_op() {
        // No stream/ring yet — must not panic or block.
        submit(48_000, &[0.1, -0.1, 0.2, -0.2]);
    }

    /// A rising ramp downsampled 2:1 (48k→24k) yields ~half the frames and stays
    /// non-decreasing — a resampling click would show as a dip.
    #[test]
    fn resample_downsample_halves_frames_and_stays_monotonic() {
        let mut state = Resampler::default();
        let n = 100;
        let input: Vec<f32> = (0..n).flat_map(|i| [i as f32, i as f32]).collect();
        let out = resample_stereo(&mut state, 48_000, 24_000, &input);
        let out_frames = (out.len() / 2) as i32;
        assert!(
            (out_frames - n / 2).abs() <= 2,
            "expected ~{} frames, got {out_frames}",
            n / 2
        );
        let lefts: Vec<f32> = out.chunks_exact(2).map(|f| f[0]).collect();
        for w in lefts.windows(2) {
            assert!(w[1] >= w[0] - 1e-3, "ramp must not dip: {w:?}");
        }
    }

    /// Two consecutive ramp buffers resampled with carried state must form one
    /// continuous rising ramp across the boundary (no gap, no click).
    #[test]
    fn resample_two_buffers_join_seamlessly() {
        let mut state = Resampler::default();
        let buf = |start: f32| -> Vec<f32> {
            (0..50)
                .flat_map(|i| [start + i as f32, start + i as f32])
                .collect()
        };
        let mut all = resample_stereo(&mut state, 48_000, 44_100, &buf(0.0));
        all.extend(resample_stereo(&mut state, 48_000, 44_100, &buf(50.0)));
        let lefts: Vec<f32> = all.chunks_exact(2).map(|f| f[0]).collect();
        assert!(
            lefts.len() > 80,
            "two 50-frame buffers should yield many frames"
        );
        for w in lefts.windows(2) {
            assert!(w[1] >= w[0] - 1e-3, "boundary must stay monotonic: {w:?}");
        }
    }

    #[test]
    fn volume_and_enable_round_trip_through_gain() {
        set_enabled(true);
        set_volume(0.5);
        assert!((gain() - 0.5).abs() < 1e-6);
        set_volume(2.0); // clamped to 1.0
        assert!((gain() - 1.0).abs() < 1e-6);
        set_enabled(false);
        assert_eq!(gain(), 0.0);
        set_enabled(true);
    }
}
