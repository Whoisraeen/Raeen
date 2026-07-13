//! Host GPU backend abstraction.
//!
//! XPS5X targets multiple host graphics APIs — Vulkan (Windows/Linux)
//! and Metal (macOS) — from a single GNM translation layer. The
//! [`GpuBackend`] trait is the seam between the API-agnostic command
//! translator and a concrete host driver, so the rest of the GPU crate
//! never hard-codes a single API.
//!
//! Each backend also declares the [`ShaderFormat`] it consumes, which
//! tells the shader recompiler whether to emit SPIR-V (Vulkan) or MSL
//! (Metal) for a given host.

use xps5x_core::error::GpuError;

/// The compiled shader representation a backend consumes.
///
/// The shader recompiler decodes PS5 RDNA2 ISA into a common IR, then
/// emits the format the active backend requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderFormat {
    /// SPIR-V bytecode, consumed by Vulkan.
    SpirV,
    /// Metal Shading Language source/bytecode, consumed by Metal.
    Msl,
}

/// Identifies a concrete host backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Vulkan 1.3 (Windows, Linux, and macOS via MoltenVK).
    Vulkan,
    /// Native Metal (macOS).
    Metal,
}

impl BackendKind {
    /// The shader format this backend consumes.
    pub fn shader_format(self) -> ShaderFormat {
        match self {
            BackendKind::Vulkan => ShaderFormat::SpirV,
            BackendKind::Metal => ShaderFormat::Msl,
        }
    }
}

/// The host GPU backend interface.
///
/// A backend owns the host graphics device and executes translated GNM
/// commands. This trait intentionally stays small at this stage — it
/// captures the lifecycle and capability queries that the command
/// translator needs to remain API-agnostic. Command submission,
/// swapchain, and pipeline methods will be added as those subsystems
/// come online.
pub trait GpuBackend {
    /// Human-readable backend name (for logs and the debug UI).
    fn name(&self) -> &'static str;

    /// Which concrete backend this is.
    fn kind(&self) -> BackendKind;

    /// The shader format this backend consumes.
    fn shader_format(&self) -> ShaderFormat {
        self.kind().shader_format()
    }

    /// Initialize the host device (instance/device/queues).
    ///
    /// # Errors
    ///
    /// Returns a [`GpuError`] if no suitable device is available or
    /// device creation fails.
    fn init(&mut self) -> Result<(), GpuError>;

    /// Whether [`GpuBackend::init`] has completed successfully.
    fn is_initialized(&self) -> bool;
}

/// Select the default backend for the host platform.
///
/// macOS prefers native Metal; every other platform uses Vulkan.
pub fn default_backend_kind() -> BackendKind {
    if cfg!(target_os = "macos") {
        BackendKind::Metal
    } else {
        BackendKind::Vulkan
    }
}

/// Create a boxed backend for the given kind.
///
/// `validation` enables API validation layers where supported (Vulkan
/// validation layers / Metal API validation).
pub fn create_backend(kind: BackendKind, validation: bool) -> Box<dyn GpuBackend> {
    match kind {
        BackendKind::Vulkan => Box::new(crate::vulkan::VulkanBackend::new(validation)),
        BackendKind::Metal => Box::new(crate::metal::MetalBackend::new(validation)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_formats_match_backend() {
        assert_eq!(BackendKind::Vulkan.shader_format(), ShaderFormat::SpirV);
        assert_eq!(BackendKind::Metal.shader_format(), ShaderFormat::Msl);
    }

    #[test]
    fn create_backend_reports_matching_kind() {
        let vk = create_backend(BackendKind::Vulkan, false);
        assert_eq!(vk.kind(), BackendKind::Vulkan);
        assert_eq!(vk.shader_format(), ShaderFormat::SpirV);
        assert!(!vk.is_initialized());

        let mtl = create_backend(BackendKind::Metal, false);
        assert_eq!(mtl.kind(), BackendKind::Metal);
        assert_eq!(mtl.shader_format(), ShaderFormat::Msl);
        assert!(!mtl.is_initialized());
    }
}
