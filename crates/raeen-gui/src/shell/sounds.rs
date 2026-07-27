//! Shell UI sound packs.
//!
//! A pack is a directory `sounds/<name>/` holding any of `move.wav`,
//! `confirm.wav`, `back.wav`, `launch.wav`. Raeen ships none — packs are
//! user-supplied, like themes and wallpapers. Clips decode once at pack
//! selection (hound), mono is widened to stereo, and playback goes through
//! [`raeen_audio::output::play_ui`], which mixes additively over guest audio
//! and respects Settings ▸ Audio volume/enable.

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The Shell events a pack can voice, in clip-array order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiSound {
    /// Focus moved (rail tile, settings row, pill, Control Center card).
    Move,
    /// Confirm/activate.
    Confirm,
    /// Back/cancel.
    Back,
    /// A game launch began.
    Launch,
}

impl UiSound {
    const ALL: [UiSound; 4] = [
        UiSound::Move,
        UiSound::Confirm,
        UiSound::Back,
        UiSound::Launch,
    ];

    fn file_name(self) -> &'static str {
        match self {
            UiSound::Move => "move.wav",
            UiSound::Confirm => "confirm.wav",
            UiSound::Back => "back.wav",
            UiSound::Launch => "launch.wav",
        }
    }

    fn index(self) -> usize {
        match self {
            UiSound::Move => 0,
            UiSound::Confirm => 1,
            UiSound::Back => 2,
            UiSound::Launch => 3,
        }
    }
}

/// One decoded clip: sample rate + interleaved-stereo f32.
type Clip = (u32, Arc<Vec<f32>>);

/// A loaded (possibly empty) UI sound pack. `"off"`, a missing directory, or
/// a directory with no decodable clips all yield the silent pack — playing is
/// then a no-op, never an error.
#[derive(Default)]
pub struct SoundPack {
    clips: [Option<Clip>; 4],
}

impl SoundPack {
    /// Load `root/<name>` (`"off"` → the silent pack). Individual clips that
    /// are missing or fail to decode are skipped with a log line naming the
    /// file; the rest of the pack still works.
    pub fn load(root: &Path, name: &str) -> Self {
        let mut pack = Self::default();
        if name == "off" || name.is_empty() {
            return pack;
        }
        let dir = root.join(name);
        for sound in UiSound::ALL {
            let path = dir.join(sound.file_name());
            if !path.is_file() {
                continue;
            }
            match decode_wav(&path) {
                Ok((rate, samples)) => {
                    // Resample once at load (rubato windowed-sinc) so playback
                    // needs no per-play linear resampling and clips sound
                    // clean on the 48 kHz output path.
                    let (rate, samples) = resample_clip_to_48k(rate, samples);
                    pack.clips[sound.index()] = Some((rate, Arc::new(samples)));
                }
                Err(err) => {
                    tracing::warn!(file = %path.display(), error = %err, "sound pack clip skipped");
                }
            }
        }
        pack
    }

    /// Whether any clip decoded — used to label a selected-but-empty pack.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.clips.iter().all(Option::is_none)
    }

    /// Queue `sound` on the host mixer. Silent no-op when the pack has no
    /// clip for it.
    pub fn play(&self, sound: UiSound) {
        if let Some((rate, samples)) = &self.clips[sound.index()] {
            raeen_audio::output::play_ui(*rate, samples);
        }
    }
}

/// Decode a WAV file to interleaved-stereo f32 at its native rate. Supports
/// 16/24/32-bit integer and 32-bit float PCM; mono is duplicated to stereo,
/// >2 channels take the first two. Clips are capped at ~5 s — a UI cue, not a
/// soundtrack.
fn decode_wav(path: &PathBuf) -> Result<(u32, Vec<f32>), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let max_frames = spec.sample_rate as usize * 5;
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .take(max_frames * channels)
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / ((1u64 << (spec.bits_per_sample - 1)) as f32);
            reader
                .samples::<i32>()
                .take(max_frames * channels)
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };

    let mut stereo = Vec::with_capacity(interleaved.len() / channels * 2);
    for frame in interleaved.chunks(channels) {
        let l = frame.first().copied().unwrap_or(0.0);
        let r = frame.get(1).copied().unwrap_or(l);
        stereo.push(l);
        stereo.push(r);
    }
    Ok((spec.sample_rate, stereo))
}

/// The host mixer's native rate; every clip is converted to this at load.
const TARGET_RATE: u32 = 48_000;

/// One-shot windowed-sinc resample (rubato) of an interleaved-stereo clip to
/// [`TARGET_RATE`]. Any setup or processing failure returns the clip
/// unchanged at its native rate — the mixer's linear fallback still plays it.
fn resample_clip_to_48k(rate: u32, stereo: Vec<f32>) -> (u32, Vec<f32>) {
    use rubato::Resampler;
    let frames = stereo.len() / 2;
    if rate == TARGET_RATE || rate == 0 || frames == 0 {
        return (rate, stereo);
    }
    const SINC_LEN: usize = 128;
    let ratio = f64::from(TARGET_RATE) / f64::from(rate);
    // Zero-pad the input past the sinc filter's group delay so the clip's
    // real ending flushes out of the filter within one process call.
    let padded = frames + SINC_LEN;
    let mut left = Vec::with_capacity(padded);
    let mut right = Vec::with_capacity(padded);
    for frame in stereo.chunks_exact(2) {
        left.push(frame[0]);
        right.push(frame[1]);
    }
    left.resize(padded, 0.0);
    right.resize(padded, 0.0);
    let params = rubato::SincInterpolationParameters {
        sinc_len: SINC_LEN,
        f_cutoff: 0.95,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let mut resampler = match rubato::SincFixedIn::<f32>::new(ratio, 1.0, params, padded, 2) {
        Ok(resampler) => resampler,
        Err(e) => {
            tracing::warn!(error = %e, "clip resampler setup failed — keeping native rate");
            return (rate, stereo);
        }
    };
    let delay = resampler.output_delay();
    let out = match resampler.process(&[left, right], None) {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!(error = %e, "clip resample failed — keeping native rate");
            return (rate, stereo);
        }
    };
    // Skip the filter latency, keep exactly the clip's resampled length.
    let expected = (frames as f64 * ratio).round() as usize;
    let available = out[0].len().min(out[1].len()).saturating_sub(delay);
    let out_frames = expected.min(available);
    let mut interleaved = Vec::with_capacity(out_frames * 2);
    for i in delay..delay + out_frames {
        interleaved.push(out[0][i]);
        interleaved.push(out[1][i]);
    }
    (TARGET_RATE, interleaved)
}

/// Selectable pack names: `"off"` plus every directory under `root`.
pub fn available_packs(root: &Path) -> Vec<String> {
    let mut packs = vec!["off".to_string()];
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut dirs: Vec<String> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        dirs.sort();
        packs.extend(dirs);
    }
    packs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, spec: hound::WavSpec, samples: &[i16]) {
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        for s in samples {
            writer.write_sample(*s).expect("write sample");
        }
        writer.finalize().expect("finalize wav");
    }

    #[test]
    fn off_and_missing_packs_are_silent_not_errors() {
        let root = Path::new("this/does/not/exist");
        assert!(SoundPack::load(root, "off").is_empty());
        assert!(SoundPack::load(root, "nope").is_empty());
        // Playing from an empty pack is a no-op.
        SoundPack::load(root, "off").play(UiSound::Confirm);
    }

    #[test]
    fn loads_mono_i16_clip_and_widens_to_stereo() {
        let base = std::env::temp_dir().join(format!("raeen-sounds-{}", std::process::id()));
        let pack_dir = base.join("testpack");
        std::fs::create_dir_all(&pack_dir).expect("mkdir pack");
        // Authored at 48 kHz already, so no resampling — samples are exact.
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        write_wav(&pack_dir.join("move.wav"), spec, &[i16::MAX, 0, i16::MIN]);

        let pack = SoundPack::load(&base, "testpack");
        assert!(!pack.is_empty());
        let (rate, samples) = pack.clips[UiSound::Move.index()]
            .as_ref()
            .expect("move.wav decoded");
        assert_eq!(*rate, 48_000);
        // Mono widened: 3 frames -> 6 samples, L == R.
        assert_eq!(samples.len(), 6);
        assert_eq!(samples[0], samples[1]);
        assert!((samples[0] - 1.0).abs() < 1e-3);
        assert!((samples[4] + 1.0).abs() < 1e-1);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn non_native_rate_clips_are_sinc_resampled_to_48k() {
        // 0.1 s of a constant tone at 44.1 kHz -> ~0.1 s at 48 kHz.
        let frames = 4_410usize;
        let stereo: Vec<f32> = std::iter::repeat_n([0.5f32, 0.5], frames)
            .flatten()
            .collect();
        let (rate, out) = resample_clip_to_48k(44_100, stereo);
        assert_eq!(rate, 48_000);
        let out_frames = out.len() / 2;
        let expected = frames * 48_000 / 44_100;
        assert!(
            out_frames.abs_diff(expected) <= expected / 10,
            "expected ~{expected} frames, got {out_frames}"
        );
        // The steady-state middle of the clip preserves the amplitude.
        let mid = out_frames / 2 * 2;
        assert!((out[mid] - 0.5).abs() < 0.05, "mid sample {}", out[mid]);
        // 48 kHz input passes through untouched.
        let passthrough = vec![0.25f32; 8];
        assert_eq!(
            resample_clip_to_48k(48_000, passthrough.clone()),
            (48_000, passthrough)
        );
    }

    #[test]
    fn available_packs_lists_off_plus_directories() {
        let base = std::env::temp_dir().join(format!("raeen-packs-{}", std::process::id()));
        std::fs::create_dir_all(base.join("alpha")).expect("mkdir");
        std::fs::create_dir_all(base.join("beta")).expect("mkdir");
        std::fs::write(base.join("stray.wav"), b"x").expect("write file");

        let packs = available_packs(&base);
        assert_eq!(packs, vec!["off", "alpha", "beta"]);

        let _ = std::fs::remove_dir_all(&base);
    }
}
