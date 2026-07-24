//! # Raeen Audio
//!
//! Emulates the PS5's Tempest 3D AudioTech engine.
//! Provides HRTF-based spatial audio processing supporting
//! up to 128 audio objects positioned in 3D space.

pub mod hrtf;
pub mod output;
pub mod pcm;
pub mod tempest;

use tracing::info;

/// Audio engine state.
pub struct AudioEngine {
    /// Whether audio is enabled.
    pub enabled: bool,
    /// Master volume (0.0 - 1.0).
    pub master_volume: f32,
    /// Whether spatial audio is active.
    pub spatial_audio: bool,
    /// Active audio voices.
    pub active_voices: u32,
}

impl AudioEngine {
    pub fn new(enabled: bool, spatial: bool) -> Self {
        info!(
            "Audio engine created (enabled={}, spatial={})",
            enabled, spatial
        );
        Self {
            enabled,
            master_volume: 1.0,
            spatial_audio: spatial,
            active_voices: 0,
        }
    }
}
