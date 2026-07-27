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
                Ok(clip) => pack.clips[sound.index()] = Some(clip),
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
fn decode_wav(path: &PathBuf) -> Result<Clip, String> {
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
    Ok((spec.sample_rate, Arc::new(stereo)))
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
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        write_wav(&pack_dir.join("move.wav"), spec, &[i16::MAX, 0, i16::MIN]);

        let pack = SoundPack::load(&base, "testpack");
        assert!(!pack.is_empty());
        let (rate, samples) = pack.clips[UiSound::Move.index()]
            .as_ref()
            .expect("move.wav decoded");
        assert_eq!(*rate, 44_100);
        // Mono widened: 3 frames -> 6 samples, L == R.
        assert_eq!(samples.len(), 6);
        assert_eq!(samples[0], samples[1]);
        assert!((samples[0] - 1.0).abs() < 1e-3);
        assert!((samples[4] + 1.0).abs() < 1e-1);

        let _ = std::fs::remove_dir_all(&base);
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
