//! Guest-PCM → interleaved-stereo f32 conversion.
//!
//! A clean-room Rust port of SharpEmu's `AudioPcmConversion`
//! (`src/SharpEmu.Libs/Audio/AudioPcmConversion.cs`, GPL-2.0-or-later), adapted
//! to emit interleaved-stereo **f32** — the format [`crate::output::submit`]
//! consumes — rather than SharpEmu's stereo s16. SharpEmu quantises to 16-bit
//! only because its host stream is s16; Raeen's cpal stream is f32, so
//! quantising here would throw away precision for nothing.
//!
//! Semantics preserved verbatim from SharpEmu:
//! - **mono** is duplicated to both output channels;
//! - **stereo / multichannel** keep the front L/R pair (a 7.1 bed is downmixed
//!   to its front pair — SharpEmu reads channel 0 and channel 1 only);
//! - the per-submission **volume** is clamped once to `[0, 1]` (it is constant
//!   for the whole buffer, so clamping per sample inside the loop would be
//!   wasted work on every real-time buffer);
//! - **float** samples are `NaN → 0` and clamped to `[-1, 1]`;
//! - the **asymmetric fixed-point scale** (SharpEmu treats `32768`, the
//!   magnitude of s16 `MIN`, as the negative full-scale) survives as the s16→f32
//!   normalisation divisor, so an s16 `-32768` maps to exactly `-1.0`. This is
//!   the f32 mirror of SharpEmu's `5a08a9b` "audio overflow crash" fix, whose
//!   float→s16 form used `scale = value < 0 ? 32768 : 32767` so that `+1.0`
//!   lands on `short.MaxValue` (32767) instead of overflowing to `+32768`.
//!
//! Deliberate, documented deviation: because the output is f32, full-scale
//! float `+1.0` maps to exactly `+1.0` here (not to `32767/32768 = 0.99997` as
//! SharpEmu's s16 quantisation would). The asymmetry is therefore only
//! observable on the s16 input path (the `/32768` divisor).
//!
//! Hardening (mirrors SharpEmu `e13cb28` "harden AudioOut2 stack out-buffer
//! writes against canary smash" and `5a08a9b`): the frame loop only reads frames
//! that fully fit inside `source`, so a malformed frame count, channel count, or
//! a short/mis-sized buffer can never read out of bounds — the caller bounds the
//! byte length before reading guest memory, and this function bounds the frame
//! count against the slice it actually received.

/// The magnitude of `i16::MIN` — SharpEmu's negative full-scale, used as the
/// s16→f32 normalisation divisor so `-32768 → -1.0` exactly.
const S16_SCALE: f32 = 32768.0;

/// Convert an interleaved guest PCM buffer (mono / stereo / multichannel, s16 or
/// float32) to interleaved-stereo f32 for the host mixer.
///
/// - `source` — the raw guest bytes (little-endian).
/// - `frames` — the number of PCM frames the caller expects; the real count is
///   capped to what `source` can hold, so an over-large `frames` is harmless.
/// - `channels` — source channels per frame (`1` mono, `2` stereo, `8` 7.1, …).
/// - `is_float` — `true` for float32 samples, `false` for signed 16-bit.
/// - `volume` — per-submission gain, clamped to `[0, 1]` (`NaN → 0`).
///
/// Returns interleaved-stereo f32 (`out.len() == 2 * processed_frames`).
#[must_use]
pub fn convert_to_stereo_f32(
    source: &[u8],
    frames: usize,
    channels: usize,
    is_float: bool,
    volume: f32,
) -> Vec<f32> {
    let channels = channels.max(1);
    let bytes_per_sample = if is_float { 4 } else { 2 };
    let source_frame_size = channels * bytes_per_sample;
    // Clamp volume once for the whole submission (SharpEmu does the same).
    let volume = if volume.is_nan() {
        0.0
    } else {
        volume.clamp(0.0, 1.0)
    };
    // Only the frames that fully fit in `source` — a bad frame count or a short
    // buffer yields fewer frames, never an out-of-bounds read. `source_frame_size`
    // is >= 2 (channels >= 1, bps >= 2), so the division is always safe.
    let available_frames = source.len() / source_frame_size;
    let frames = frames.min(available_frames);
    let mut out = Vec::with_capacity(frames * 2);
    for f in 0..frames {
        let base = f * source_frame_size;
        let left = read_sample(source, base, 0, bytes_per_sample, is_float);
        // Mono duplicates channel 0; anything wider keeps the front pair.
        let right = if channels == 1 {
            left
        } else {
            read_sample(source, base, 1, bytes_per_sample, is_float)
        };
        out.push(apply_volume(left, volume));
        out.push(apply_volume(right, volume));
    }
    out
}

/// Read one interleaved sample as a normalised f32 in roughly `[-1, 1]`.
///
/// `frame_base + channel * bps + bps <= source.len()` is guaranteed by the
/// caller's `available_frames` bound (channel is only ever 0 or 1, and for a
/// >= 2-channel frame both lie inside `source_frame_size`).
fn read_sample(src: &[u8], frame_base: usize, channel: usize, bps: usize, is_float: bool) -> f32 {
    let o = frame_base + channel * bps;
    if is_float {
        convert_float_sample(f32::from_le_bytes([
            src[o],
            src[o + 1],
            src[o + 2],
            src[o + 3],
        ]))
    } else {
        // Asymmetric normalisation: divide by 32768 (the magnitude of s16 MIN),
        // so -32768 → -1.0 exactly. Matches SharpEmu treating 32768 as the
        // negative full-scale.
        f32::from(i16::from_le_bytes([src[o], src[o + 1]])) / S16_SCALE
    }
}

/// SharpEmu `ConvertFloatSample` in f32 form: `NaN → 0`, otherwise clamp to
/// `[-1, 1]`. (The s16 quantisation SharpEmu applied afterwards is dropped —
/// the output stays float.)
fn convert_float_sample(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(-1.0, 1.0)
    }
}

/// Apply the pre-clamped `[0, 1]` volume; a final clamp keeps the mixed result
/// in range (SharpEmu clamps its post-volume s16 to the short range for the
/// same reason).
fn apply_volume(sample: f32, volume: f32) -> f32 {
    (sample * volume).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(v: f32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn s16_bytes(v: i16) -> [u8; 2] {
        v.to_le_bytes()
    }

    /// Mirrors SharpEmu `AudioPcmConversionTests.FloatFullScaleMapsToSignedPcmEndpoints`,
    /// adapted to f32 output: full-scale float `-1.0`/`+1.0` map to the f32
    /// endpoints `-1.0`/`+1.0` (SharpEmu's s16 `MIN`/`MAX`).
    #[test]
    fn float_full_scale_maps_to_f32_endpoints() {
        let mut source = Vec::new();
        source.extend(f32_bytes(-1.0));
        source.extend(f32_bytes(1.0));
        let out = convert_to_stereo_f32(&source, 1, 2, true, 1.0);
        assert_eq!(out, vec![-1.0, 1.0]);
    }

    /// Mirrors SharpEmu `AudioPcmConversionTests.FloatNaNMapsToSilence`: a NaN
    /// float sample becomes silence on both channels.
    #[test]
    fn float_nan_maps_to_silence() {
        let source = f32_bytes(f32::NAN);
        let out = convert_to_stereo_f32(&source, 1, 1, true, 1.0);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn s16_stereo_normalises_front_pair() {
        let mut source = Vec::new();
        source.extend(s16_bytes(16_384)); // +0.5
        source.extend(s16_bytes(-16_384)); // -0.5
        let out = convert_to_stereo_f32(&source, 1, 2, false, 1.0);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-4);
        assert!((out[1] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn s16_min_maps_to_exactly_negative_one() {
        // The asymmetric divisor (32768) makes s16 MIN land on exactly -1.0.
        let out = convert_to_stereo_f32(&s16_bytes(i16::MIN), 1, 1, false, 1.0);
        assert_eq!(out, vec![-1.0, -1.0]);
    }

    #[test]
    fn s16_mono_duplicates_to_both_channels() {
        let out = convert_to_stereo_f32(&s16_bytes(32_767), 1, 1, false, 1.0);
        assert_eq!(out.len(), 2);
        assert!((out[0] - out[1]).abs() < 1e-9);
        assert!(out[0] > 0.99);
    }

    #[test]
    fn float_7_1_keeps_front_pair_only() {
        // 8-channel float frame: only channel 0 and channel 1 reach the output.
        let mut source = Vec::new();
        for i in 0..8 {
            source.extend(f32_bytes(i as f32 * 0.1));
        }
        let out = convert_to_stereo_f32(&source, 1, 8, true, 1.0);
        assert_eq!(out.len(), 2);
        assert!(out[0].abs() < 1e-6); // ch0 = 0.0
        assert!((out[1] - 0.1).abs() < 1e-6); // ch1 = 0.1
    }

    #[test]
    fn volume_scales_and_clamps() {
        let mut stereo = Vec::new();
        stereo.extend(f32_bytes(0.8));
        stereo.extend(f32_bytes(-0.8));

        // Half volume halves the amplitude.
        let out = convert_to_stereo_f32(&stereo, 1, 2, true, 0.5);
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] + 0.4).abs() < 1e-6);

        // Volume > 1 is clamped to 1 (no gain past unity).
        let out = convert_to_stereo_f32(&stereo, 1, 2, true, 4.0);
        assert!((out[0] - 0.8).abs() < 1e-6);

        // NaN volume is treated as silence.
        let out = convert_to_stereo_f32(&stereo, 1, 2, true, f32::NAN);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn float_out_of_range_input_is_clamped() {
        let mut source = Vec::new();
        source.extend(f32_bytes(2.5)); // over-range positive
        source.extend(f32_bytes(-3.0)); // over-range negative
        let out = convert_to_stereo_f32(&source, 1, 2, true, 1.0);
        assert_eq!(out, vec![1.0, -1.0]);
    }

    #[test]
    fn overlong_frame_count_is_bounded_to_the_buffer() {
        // Buffer holds exactly one stereo-f32 frame; a caller claiming 1000
        // frames must not read past it — only the one real frame is produced.
        let mut source = Vec::new();
        source.extend(f32_bytes(0.25));
        source.extend(f32_bytes(-0.25));
        let out = convert_to_stereo_f32(&source, 1000, 2, true, 1.0);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn empty_or_short_buffer_yields_no_frames_without_panicking() {
        assert!(convert_to_stereo_f32(&[], 4, 2, true, 1.0).is_empty());
        // Three bytes is less than one s16-stereo frame (4 bytes): no output.
        assert!(convert_to_stereo_f32(&[1, 2, 3], 4, 2, false, 1.0).is_empty());
    }
}
