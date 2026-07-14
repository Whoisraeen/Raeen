//! Runtime configuration for XPS5X.
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
    /// Name of the active Shell theme (spec `2026-07-13-xps5x-shell-design.md`
    /// §6/§10). SM2a only ships the in-code default theme, so this is a
    /// single-item selector for now; SM2b's on-disk theme loader is what
    /// actually resolves this name to a `themes/<name>` directory.
    pub selected_theme: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            fullscreen: true,
            window_width: 1920,
            window_height: 1080,
            vsync: true,
            selected_theme: "default".to_string(),
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
    /// GPU device index to use (0 = auto/default).
    pub gpu_device_index: u32,
    /// Enable GPU validation layers (debug only).
    pub validation_layers: bool,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Vulkan,
            resolution_scale: 1.0,
            shader_cache: true,
            gpu_device_index: 0,
            validation_layers: false,
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
    /// Enable DualSense-specific features (haptics, adaptive triggers).
    pub dualsense_features: bool,
    /// Controller deadzone (0.0 - 1.0).
    pub deadzone: f32,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            dualsense_features: true,
            deadzone: 0.15,
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
    /// Enable GPU command stream dumping.
    pub dump_gpu_commands: bool,
    /// Enable shader disassembly output.
    pub dump_shaders: bool,
    /// Enable syscall tracing.
    pub trace_syscalls: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            logging: true,
            log_level: "info".to_string(),
            dump_gpu_commands: false,
            dump_shaders: false,
            trace_syscalls: false,
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
