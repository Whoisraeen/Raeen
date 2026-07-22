//! Vulkan backend — host GPU interface.
//!
//! Owns the Vulkan instance, physical/logical device, queue, and command pool,
//! and executes translated GNM/AGC commands against them.
//!
//! ## Status
//!
//! - [`instance`] — real Vulkan 1.3 bring-up.
//! - [`offscreen`] — offscreen draw + pixel readback.
//! - [`shaders`] — hand-built SPIR-V for the backend smoke test.
//! - M2 path — AGC PM4 → [`crate::agc_exec`] → [`VulkanBackend::render_m2_triangle`]
//!   using `kyty-graphics` SPIR-V (see [`crate::shader_bridge`]).
//!
//! Swapchain presentation via `libSceVideoOut` remains M3.

pub(crate) mod cache;
pub mod compute;
pub mod instance;
pub mod offscreen;
pub mod shaders;

use crate::backend::{BackendKind, GpuBackend};
use crate::shader_bridge;
use instance::VulkanDevice;
use raeen_core::error::GpuError;
use tracing::info;

pub use cache::DrawCacheStats;
pub use instance::validation_error_count;
pub use offscreen::{
    CLEAR_COLOR, IndexBinding, RenderedImage, render_triangle, render_triangle_with_spirv, unorm8,
};
pub use shaders::TRIANGLE_COLOR;

/// Vulkan 1.3 backend.
///
/// [`GpuBackend::init`] performs the real device bring-up; before that the
/// backend owns no GPU resources.
pub struct VulkanBackend {
    /// Whether validation layers were requested (they are enabled only if the
    /// layer is actually installed — see [`VulkanDevice::validation_enabled`]).
    pub validation: bool,
    /// The live device, present only after a successful [`GpuBackend::init`].
    device: Option<VulkanDevice>,
}

impl VulkanBackend {
    pub fn new(validation: bool) -> Self {
        info!("Vulkan backend created (validation={validation})");
        Self {
            validation,
            device: None,
        }
    }

    /// The live device, or `None` before a successful `init`.
    pub fn device(&self) -> Option<&VulkanDevice> {
        self.device.as_ref()
    }

    /// Render one triangle offscreen and read the pixels back (hand SPIR-V).
    ///
    /// # Errors
    ///
    /// [`GpuError::VulkanInitFailed`] if the backend has not been initialized,
    /// or if any Vulkan operation in the draw fails.
    pub fn render_test_triangle(&self, width: u32, height: u32) -> Result<RenderedImage, GpuError> {
        let device = self.device.as_ref().ok_or_else(|| {
            GpuError::VulkanInitFailed("backend not initialized — call init() first".to_owned())
        })?;
        render_triangle(device, width, height)
    }

    /// Cache-effectiveness counters (stage A instrumentation), or `None`
    /// before `init`. See [`DrawCacheStats`].
    pub fn draw_cache_stats(&self) -> Option<DrawCacheStats> {
        self.device.as_ref().map(VulkanDevice::draw_cache_stats)
    }

    /// M2 draw: offscreen triangle using `kyty-graphics` SPIR-V.
    pub fn render_m2_triangle(&self, width: u32, height: u32) -> Result<RenderedImage, GpuError> {
        let device = self.device.as_ref().ok_or_else(|| {
            GpuError::VulkanInitFailed("backend not initialized — call init() first".to_owned())
        })?;
        let (vs, fs) = shader_bridge::m2_triangle_spirv()?;
        render_triangle_with_spirv(device, width, height, &vs, &fs)
    }
}

impl GpuBackend for VulkanBackend {
    fn name(&self) -> &'static str {
        "Vulkan"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn init(&mut self) -> Result<(), GpuError> {
        if self.device.is_some() {
            return Ok(());
        }
        self.device = Some(VulkanDevice::new(self.validation)?);
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.device.is_some()
    }
}
