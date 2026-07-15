//! Vulkan backend — host GPU interface.
//!
//! Owns the Vulkan instance, physical/logical device, queue, and command pool,
//! and executes translated GNM commands against them.
//!
//! ## Status
//!
//! - [`instance`] — real Vulkan 1.3 bring-up (instance, device selection with
//!   `dynamicRendering`, queue, command pool, optional validation layer).
//! - [`offscreen`] — a real offscreen draw: pipeline from SPIR-V, one triangle
//!   rasterized into an `R8G8B8A8_UNORM` image, copied back to host memory.
//!   Verified by pixel readback in `tests/vulkan_triangle.rs`.
//! - [`shaders`] — hand-built SPIR-V for that triangle. The GCN→SPIR-V
//!   translation in [`crate::shader`] is **not** wired in here yet; see the
//!   module docs there for what remains.
//!
//! Still missing for a full M2 claim: PM4 command streams driving this draw
//! (rather than a hardcoded triangle), and swapchain presentation via
//! `libSceVideoOut`. See `docs/reference-port-ledger.md`.

pub mod instance;
pub mod offscreen;
pub mod shaders;

use crate::backend::{BackendKind, GpuBackend};
use instance::VulkanDevice;
use tracing::info;
use xps5x_core::error::GpuError;

pub use instance::validation_error_count;
pub use offscreen::{CLEAR_COLOR, RenderedImage, render_triangle, unorm8};
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

    /// Render one triangle offscreen and read the pixels back.
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
