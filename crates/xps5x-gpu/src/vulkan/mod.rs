//! Vulkan backend — host GPU interface.
//!
//! Manages the Vulkan instance, device, swapchain, and command
//! submission. Translated GNM commands are executed through this backend.

pub mod instance;

use crate::backend::{BackendKind, GpuBackend};
use tracing::info;
use xps5x_core::error::GpuError;

/// Placeholder Vulkan backend.
/// Full implementation will use ash (Vulkan bindings) for:
/// - Instance/device creation with validation layers
/// - Swapchain management
/// - Command buffer recording and submission
/// - Pipeline state management
/// - Descriptor set layout and management
/// - Memory allocation (via gpu-allocator)
pub struct VulkanBackend {
    /// Whether validation layers are enabled.
    pub validation: bool,
    /// Whether the backend has been initialized.
    pub initialized: bool,
}

impl VulkanBackend {
    pub fn new(validation: bool) -> Self {
        info!("Vulkan backend created (validation={})", validation);
        Self {
            validation,
            initialized: false,
        }
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
        // TODO: create VkInstance, select physical device, create logical
        // device + queues via ash. See vulkan::instance.
        info!("Vulkan backend init (validation={})", self.validation);
        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}
