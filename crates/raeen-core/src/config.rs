//! Runtime configuration for Raeen.
//!
//! Configuration is loaded from a TOML file and can be overridden
//! by command-line arguments or environment variables.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level emulator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct EmulatorConfig {
    /// General settings.
    pub general: GeneralConfig,
    /// Graphics / GPU settings.
    pub graphics: GraphicsConfig,
    /// Audio settings.
    pub audio: AudioConfig,
    /// Input / controller settings.
    pub input: InputConfig,
    /// Debug settings.
    pub debug: DebugConfig,
    /// Path settings.
    pub paths: PathConfig,
}

/// General emulator settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    /// Whether to run as a borderless fullscreen shell (the default,
    /// console-style experience) or a normal window sized by
    /// `window_width`/`window_height`.
    pub fullscreen: bool,
    /// Window width (when not fullscreen).
    pub window_width: u32,
    /// Window height (when not fullscreen).
    pub window_height: u32,
    /// Enable VSync.
    pub vsync: bool,
    /// Name of the active Shell theme (spec `2026-07-13-raeen-shell-design.md`
    /// §6/§10). SM2a only ships the in-code default theme, so this is a
    /// single-item selector for now; SM2b's on-disk theme loader is what
    /// actually resolves this name to a `themes/<name>` directory.
    pub selected_theme: String,
    /// Custom Home wallpaper: an image file name under `wallpapers/`, or
    /// `"off"` to use the active theme's own background (if any). A wallpaper
    /// overrides the theme background without editing the theme.
    pub wallpaper: String,
    /// UI sound pack: a directory name under `sounds/` holding `move.wav`,
    /// `confirm.wav`, `back.wav`, `launch.wav` (all optional), or `"off"`
    /// for a silent shell.
    pub sound_pack: String,
    /// Show the in-session performance HUD overlay (FPS, frame time,
    /// upload/present timing). Toggled in Settings ▸ Advanced or with F3.
    pub perf_hud: bool,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            fullscreen: true,
            window_width: 1920,
            window_height: 1080,
            vsync: true,
            selected_theme: "default".to_string(),
            wallpaper: "off".to_string(),
            sound_pack: "off".to_string(),
            perf_hud: false,
        }
    }
}

/// Graphics and GPU configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphicsConfig {
    /// GPU backend to use.
    pub backend: GpuBackend,
    /// Resolution scaling factor (1.0 = native PS5, 2.0 = 4K upscale).
    pub resolution_scale: f32,
    /// Enable shader cache on disk.
    pub shader_cache: bool,
    /// GPU device index to select (0-based). Out-of-range falls back to the
    /// best-scored device. Drives Vulkan physical-device selection.
    pub gpu_device_index: u32,
    /// Enable GPU validation layers (debug only).
    pub validation_layers: bool,
    /// Target frame-pacing rate in Hz — the guest vblank cadence, clamped to
    /// 24–480 (60 = native PS5). Drives `RAEEN_VBLANK_HZ`.
    pub frame_limit: u32,
    /// Present-path plugin (upscaler / frame generator) to apply, by name.
    /// `"off"` is the zero-cost identity path. Other names come from
    /// `raeen_gpu::AgcGpuSession::present_plugins()` — the built-in
    /// vendor-neutral plugins plus any user-supplied (BYO) plugin registered at
    /// startup. Raeen ships/fetches no proprietary plugin (see `plugins/`).
    pub upscaler: String,
    /// Present-time upscale factor the active upscaler targets (1.0 = native).
    /// Distinct from `resolution_scale`, which scales the guest draws.
    pub present_upscale: f32,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Vulkan,
            resolution_scale: 1.0,
            shader_cache: true,
            gpu_device_index: 0,
            validation_layers: false,
            frame_limit: 60,
            upscaler: "off".to_string(),
            present_upscale: 1.0,
        }
    }
}

/// Supported GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// Vulkan 1.3 — primary backend (Windows, Linux, macOS via MoltenVK).
    Vulkan,
    /// Metal — native macOS backend (future).
    Metal,
}

/// Audio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Enable audio output.
    pub enabled: bool,
    /// Master volume (0.0 - 1.0).
    pub volume: f32,
    /// Enable 3D spatial audio (Tempest emulation).
    pub spatial_audio: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            volume: 1.0,
            spatial_audio: true,
        }
    }
}

/// Input / controller configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InputConfig {
    /// Enable DualSense-specific features. Today this gates game vibration
    /// (rumble) passthrough to the physical controller — DualSense HID output
    /// reports or XInput — with advanced haptics / adaptive triggers to come.
    pub dualsense_features: bool,
    /// Controller deadzone (0.0 - 1.0).
    pub deadzone: f32,
    /// Button legend used by the Shell for confirm/back prompts.
    pub controller_icon_style: ControllerIconStyle,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            dualsense_features: true,
            deadzone: 0.15,
            controller_icon_style: ControllerIconStyle::PlayStation,
        }
    }
}

/// Controller button-label family shown by the Shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ControllerIconStyle {
    #[default]
    PlayStation,
    Xbox,
    Generic,
}

impl ControllerIconStyle {
    pub const fn label(self) -> &'static str {
        match self {
            Self::PlayStation => "PlayStation",
            Self::Xbox => "Xbox",
            Self::Generic => "Generic / third-party",
        }
    }

    pub const fn cycle(self, delta: i32) -> Self {
        let index = match self {
            Self::PlayStation => 0,
            Self::Xbox => 1,
            Self::Generic => 2,
        };
        match (index + delta).rem_euclid(3) {
            0 => Self::PlayStation,
            1 => Self::Xbox,
            _ => Self::Generic,
        }
    }

    pub const fn confirm(self) -> &'static str {
        match self {
            Self::PlayStation => "Cross",
            Self::Xbox => "A",
            Self::Generic => "1",
        }
    }

    pub const fn back(self) -> &'static str {
        match self {
            Self::PlayStation => "Circle",
            Self::Xbox => "B",
            Self::Generic => "2",
        }
    }
}

/// Debug / development configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Enable debug logging.
    pub logging: bool,
    /// Log level filter (trace, debug, info, warn, error).
    pub log_level: String,
    /// Dump per-draw GPU resources (vertex/index/resource buffers) to disk.
    /// Drives `RAEEN_DUMP_GPU_RESOURCES`.
    pub dump_gpu_commands: bool,
    /// Dump every distinct fetched guest shader (`.sb`/`.spv`) to disk. Drives
    /// `RAEEN_DUMP_SHADERS`.
    pub dump_shaders: bool,
    /// Trace every HLE call (very verbose). Drives `RAEEN_TRACE_HLE`.
    pub trace_syscalls: bool,
    /// Dump each presented frame to disk (PPM). Drives `RAEEN_DUMP_FRAMES`.
    pub dump_frames: bool,
    /// Log per-HLE-function call counts, boot vs steady state. Drives
    /// `RAEEN_CALL_STATS`.
    pub call_stats: bool,
    /// Periodically dump every guest thread's recent calls/backtrace when a
    /// title stalls. Drives `RAEEN_STALL_DUMP`.
    pub stall_dump: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            logging: true,
            log_level: "info".to_string(),
            dump_gpu_commands: false,
            dump_shaders: false,
            trace_syscalls: false,
            dump_frames: false,
            call_stats: false,
            stall_dump: false,
        }
    }
}

/// Path configuration for game data, firmware, and save files.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PathConfig {
    /// Directory containing game PKG/ELF files.
    pub games_dir: PathBuf,
    /// Directory for PS5 firmware modules.
    pub firmware_dir: PathBuf,
    /// Directory for save data.
    pub save_dir: PathBuf,
    /// Directory for shader cache.
    pub shader_cache_dir: PathBuf,
    /// Directory for log files.
    pub log_dir: PathBuf,
    /// User game-library folders the Shell scans for titles (Settings ▸
    /// Game Folders). May contain zero, one, or many entries; `games_dir`
    /// above remains the loader/kernel's own single game-data root and is
    /// untouched by the Shell's folder list.
    pub game_folders: Vec<PathBuf>,
    /// Path to the user's KeyProvider file (the firmware-decryption seam).
    /// The Shell only stores and displays this path — it never reads,
    /// parses, or otherwise handles key material itself.
    pub key_provider_path: PathBuf,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            games_dir: PathBuf::from("games"),
            firmware_dir: PathBuf::from("firmware"),
            save_dir: PathBuf::from("savedata"),
            shader_cache_dir: PathBuf::from("shader_cache"),
            log_dir: PathBuf::from("logs"),
            game_folders: vec![PathBuf::from("Games")],
            key_provider_path: PathBuf::new(),
        }
    }
}

impl EmulatorConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        if path.exists() {
            let contents = std::fs::read_to_string(path)?;
            let config: Self = toml::from_str(&contents)?;
            Ok(config)
        } else {
            // Return default config and write it to disk for the user.
            let config = Self::default();
            config.save(path)?;
            Ok(config)
        }
    }

    /// Save configuration to a TOML file.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let contents = toml::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_config_defaults_controller_icon_style() {
        let config: EmulatorConfig = toml::from_str("[input]\ndeadzone = 0.2\n").unwrap();
        assert_eq!(
            config.input.controller_icon_style,
            ControllerIconStyle::PlayStation
        );
    }

    #[test]
    fn old_config_defaults_perf_hud_off_and_round_trips() {
        // A config written before the HUD existed must load with it off.
        let config: EmulatorConfig = toml::from_str("[general]\nvsync = true\n").unwrap();
        assert!(!config.general.perf_hud);
        // And the toggle persists through a save/load cycle.
        let mut config = EmulatorConfig::default();
        config.general.perf_hud = true;
        let encoded = toml::to_string(&config).unwrap();
        let decoded: EmulatorConfig = toml::from_str(&encoded).unwrap();
        assert!(decoded.general.perf_hud);
    }

    /// Settings ▸ Controllers ▸ DualSense Features (the vibration/rumble
    /// routing gate) persists like every other setting: ON by default, an
    /// explicit OFF survives a save/load cycle, and a config written before
    /// the toggle existed defaults it ON.
    #[test]
    fn dualsense_features_defaults_on_and_round_trips() {
        assert!(EmulatorConfig::default().input.dualsense_features);
        let old: EmulatorConfig = toml::from_str("[input]\ndeadzone = 0.2\n").unwrap();
        assert!(old.input.dualsense_features);

        let mut config = EmulatorConfig::default();
        config.input.dualsense_features = false;
        let encoded = toml::to_string(&config).unwrap();
        let decoded: EmulatorConfig = toml::from_str(&encoded).unwrap();
        assert!(!decoded.input.dualsense_features);
    }

    #[test]
    fn controller_icon_style_round_trips_and_cycles() {
        let mut config = EmulatorConfig::default();
        config.input.controller_icon_style = ControllerIconStyle::Xbox;
        let encoded = toml::to_string(&config).unwrap();
        let decoded: EmulatorConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.input.controller_icon_style,
            ControllerIconStyle::Xbox
        );
        assert_eq!(
            ControllerIconStyle::Xbox.cycle(1),
            ControllerIconStyle::Generic
        );
        assert_eq!(
            ControllerIconStyle::PlayStation.cycle(-1),
            ControllerIconStyle::Generic
        );
    }
}
