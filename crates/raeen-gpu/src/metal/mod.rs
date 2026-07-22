//! Metal backend — native macOS host GPU interface (stub).
//!
//! On macOS, Raeen targets Apple's Metal API directly rather than going
//! through MoltenVK, to minimize translation overhead. This module is a
//! placeholder that satisfies the [`GpuBackend`](crate::backend::GpuBackend)
//! seam so the rest of the GPU crate can be written API-agnostically today.
//!
//! The real implementation will (behind `#[cfg(target_os = "macos")]`,
//! using the `metal` / `objc2` crates):
//! - Acquire the system `MTLDevice` and a command queue
//! - Manage a `CAMetalLayer` swapchain
//! - Build `MTLRenderPipelineState` / `MTLComputePipelineState` objects
//! - Consume MSL emitted by the shader recompiler (see `ShaderFormat::Msl`)

use crate::backend::{BackendKind, GpuBackend};
use raeen_core::error::GpuError;
use tracing::info;

/// Placeholder Metal backend.
pub struct MetalBackend {
    /// Whether Metal API validation is enabled.
    pub validation: bool,
    /// Whether the backend has been initialized.
    pub initialized: bool,
}

impl MetalBackend {
    pub fn new(validation: bool) -> Self {
        info!("Metal backend created (validation={})", validation);
        Self {
            validation,
            initialized: false,
        }
    }
}

impl GpuBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "Metal"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn init(&mut self) -> Result<(), GpuError> {
        // Metal is only available on macOS hosts.
        if !cfg!(target_os = "macos") {
            return Err(GpuError::MetalInitFailed(
                "Metal backend is only available on macOS".to_string(),
            ));
        }
        // TODO: MTLCreateSystemDefaultDevice(), command queue, CAMetalLayer.
        info!("Metal backend init (validation={})", self.validation);
        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}
