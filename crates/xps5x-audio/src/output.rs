//! Host audio output (WASAPI / PulseAudio / CoreAudio via cpal).
//!
//! The guest's `sceAudioOutOutput` (HLE `libSceAudioOut`) decodes its PCM
//! buffer to interleaved-stereo f32 and calls [`submit`]; a cpal output stream
//! on a dedicated thread drains a shared ring buffer to the speakers, applying
//! the user's master volume / enable from Settings ▸ Audio ([`set_volume`] /
//! [`set_enabled`]).
//!
//! Sample-rate matching is best-effort: the stream is opened at the guest's
//! 48 kHz when the device supports it (so no resampling is needed), otherwise at
//! the device default (with a small possible pitch drift, logged once). A
//! missing output device, or a device that won't do f32, is never fatal — audio
//! just stays silent.

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

/// ~250 ms of stereo at 48 kHz. Submissions beyond this drop the oldest
/// samples, so a slow/stalled consumer can never grow memory without bound.
const MAX_RING_SAMPLES: usize = 48_000 * 2 / 4;

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
            info!(
                "host audio output ready ({rate} Hz, {channels}ch; guest is 48000 Hz — minor pitch drift possible)"
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

/// Submit interleaved-stereo f32 samples from the guest (already downmixed by
/// the HLE). Drops the oldest samples if the consumer is behind so memory stays
/// bounded. A no-op until [`init`], or when muted.
pub fn submit(samples: &[f32]) {
    let Some(ring) = RING.get() else {
        return;
    };
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut ring = ring
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ring.extend(samples.iter().copied());
    if ring.len() > MAX_RING_SAMPLES {
        let excess = ring.len() - MAX_RING_SAMPLES;
        ring.drain(0..excess);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_before_init_is_a_silent_no_op() {
        // No stream/ring yet — must not panic or block.
        submit(&[0.1, -0.1, 0.2, -0.2]);
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
