//! Vulkan instance, physical-device selection, logical device, and command pool.
//!
//! This is the real host-GPU bring-up that backs [`VulkanBackend`](super::VulkanBackend).
//! It creates a Vulkan 1.3 instance (optionally with the Khronos validation
//! layer and a `VK_EXT_debug_utils` messenger routed into `tracing`), picks a
//! physical device that can do dynamic rendering, and creates a logical device
//! with one graphics queue plus a command pool.
//!
//! No surface/swapchain is involved: XPS5X renders offscreen first (see
//! [`super::offscreen`]) so the draw path is testable headlessly. Presentation
//! is wired later, when `libSceVideoOut` flips reach the backend.

use super::cache::{DrawCacheStats, DrawCaches};
use ash::{Device, Entry, Instance, ext::debug_utils, vk};
use parking_lot::{Mutex, MutexGuard};
use std::ffi::{CStr, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};
use xps5x_core::error::GpuError;

/// Count of validation-layer messages at ERROR severity, since process start.
///
/// Validation messages arrive on a plain `extern "system"` callback with no
/// per-device context, so this is necessarily a process-global counter. It
/// exists because messages otherwise go only to `tracing`, and a consumer with
/// no subscriber installed (a test binary, say) would silently discard them —
/// letting a draw that Vulkan considers invalid still look like it passed.
static VALIDATION_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Number of validation errors reported so far.
///
/// Only ever nonzero when a [`VulkanDevice`] was built with validation active.
/// Tests assert this is zero after exercising the GPU.
pub fn validation_error_count() -> u64 {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

/// The Khronos validation layer, enabled when `validation` is requested *and*
/// the layer is actually installed (it ships with the Vulkan SDK, so it is
/// absent on most end-user machines and on CI).
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// A live Vulkan device: instance, physical device, logical device, queue, pool.
///
/// Dropping this destroys every handle it owns, in reverse creation order,
/// after waiting for the device to go idle.
pub struct VulkanDevice {
    /// The dynamically-loaded Vulkan loader.
    ///
    /// Never read directly, but it owns the `vkGetInstanceProcAddr` the
    /// `Instance`/`Device` function pointers were loaded from, so it must
    /// outlive them. Dropping it early would invalidate every other handle.
    #[allow(dead_code)]
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    /// Driver-side cache of compiled pipeline binaries, reused across draws.
    pipeline_cache: vk::PipelineCache,
    /// Application-side caches of long-lived draw/dispatch resources
    /// (pipelines, layouts, shader modules, persistent render targets,
    /// command buffer/fence, descriptor pool). See [`super::cache`] for the
    /// inventory and the locking/thread-ownership contract: the mutex is held
    /// for a whole synchronous draw or dispatch, which serializes every user
    /// of the cached resources and of the queue.
    caches: Mutex<DrawCaches>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Debug messenger, present only when validation is active.
    debug: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    /// Human-readable name of the selected physical device.
    device_name: String,
    /// Whether `VK_EXT_depth_range_unrestricted` was enabled on the logical
    /// device. When false, viewport min/max depth must stay ordered within
    /// [0, 1] — a guest reverse-Z range cannot be honoured.
    depth_range_unrestricted: bool,
    /// Whether the validation layer was actually enabled.
    validation_enabled: bool,
}

impl VulkanDevice {
    /// Bring up a Vulkan 1.3 device.
    ///
    /// `validation` requests the Khronos validation layer; if it is not
    /// installed the device is still created, just without it (a warning is
    /// logged). This keeps the same code path working on developer machines
    /// with the SDK and on bare CI runners.
    ///
    /// # Errors
    ///
    /// [`GpuError::VulkanInitFailed`] if no loader/driver is present or a
    /// Vulkan call fails; [`GpuError::NoSuitableDevice`] if no physical device
    /// supports Vulkan 1.3 dynamic rendering with a graphics queue.
    pub fn new(validation: bool) -> Result<Self, GpuError> {
        // SAFETY: `Entry::load` dlopen's the platform Vulkan loader
        // (vulkan-1.dll / libvulkan.so) and resolves `vkGetInstanceProcAddr`.
        // It is unsafe because it runs the loader's initialization code; a
        // missing/!broken loader is reported as an `Err`, not UB. The returned
        // `Entry` is stored in `self.entry` and thus outlives every handle
        // derived from it.
        let entry = unsafe { Entry::load() }
            .map_err(|e| GpuError::VulkanInitFailed(format!("Vulkan loader unavailable: {e}")))?;

        let (instance, validation_enabled) = Self::create_instance(&entry, validation)?;

        // Attach the debug messenger before device selection so that
        // validation errors during the remaining setup are still reported.
        let debug = if validation_enabled {
            Self::create_debug_messenger(&entry, &instance)
        } else {
            None
        };

        // From here on, any early return must not leak the instance, so each
        // fallible step cleans up explicitly via `destroy_partial`.
        let result = Self::pick_and_create_device(&entry, &instance);
        let (physical_device, device, queue_family_index, device_name, depth_range_unrestricted) =
            match result {
                Ok(v) => v,
                Err(e) => {
                    Self::destroy_partial(&instance, debug);
                    return Err(e);
                }
            };

        // SAFETY: `queue_family_index` came from the queue-family enumeration
        // for `physical_device` and was requested in `device`'s create info
        // with queue count 1, so index 0 exists.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // SAFETY: `physical_device` is a valid handle from this instance; the
        // call only fills the returned properties struct.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: `device` is valid and `pool_info` references only the queue
        // family index validated above. No allocator callbacks are supplied.
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: `device` was created above and no handles were made
                // from it yet, so it can be destroyed directly.
                unsafe { device.destroy_device(None) };
                Self::destroy_partial(&instance, debug);
                return Err(GpuError::VulkanInitFailed(format!(
                    "vkCreateCommandPool failed: {e}"
                )));
            }
        };

        // A process-wide pipeline cache so the driver reuses compiled shader
        // binaries across draws. A title re-binds the same pipelines thousands
        // of times a frame; without this each rebind recompiles from SPIR-V.
        // A failed cache is non-fatal — fall back to no cache.
        // SAFETY: `device` is valid; the default create-info is inert.
        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
                .unwrap_or(vk::PipelineCache::null());

        info!(
            "Vulkan device ready: {device_name} (validation={validation_enabled}, depth_range_unrestricted={depth_range_unrestricted}, graphics queue family {queue_family_index})"
        );

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            pipeline_cache,
            caches: Mutex::new(DrawCaches::default()),
            memory_properties,
            debug,
            device_name,
            depth_range_unrestricted,
            validation_enabled,
        })
    }

    /// Name of the selected physical device, e.g. `NVIDIA GeForce RTX 4070`.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether `VK_EXT_depth_range_unrestricted` is active. A PS5 title's
    /// reverse-Z viewport (`zoffset + zscale < zoffset`, e.g. [1, 0]) is only
    /// expressible when this is true; the depth path must name the failure
    /// otherwise instead of silently clamping.
    pub fn depth_range_unrestricted(&self) -> bool {
        self.depth_range_unrestricted
    }

    /// Shared pipeline cache; pass to `create_graphics_pipelines` so repeated
    /// pipelines reuse the driver's compiled binaries. May be null if creation
    /// failed — that is still a valid argument.
    pub(crate) fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    /// The long-lived draw/dispatch resource caches (stage A).
    ///
    /// The returned guard must be held for the entire draw/dispatch that uses
    /// any cached handle — the synchronous fence wait happens under it, which
    /// is what makes reusing the command buffer, fence, and descriptor pool
    /// sound (see [`super::cache`] module docs).
    pub(crate) fn draw_caches(&self) -> MutexGuard<'_, DrawCaches> {
        self.caches.lock()
    }

    /// Cache-effectiveness counters, cumulative since device creation.
    ///
    /// This is the stage A instrumentation: a title run (or test) reads it to
    /// prove that repeated draws stop rebuilding pipelines and render targets.
    pub fn draw_cache_stats(&self) -> DrawCacheStats {
        self.caches.lock().stats
    }

    /// Whether the Khronos validation layer is actually active.
    pub fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// The selected physical device.
    ///
    /// Needed by the presentation path to query surface capabilities/formats.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// The graphics queue family index [`Self::queue`] was taken from.
    ///
    /// Needed by the presentation path to check surface support on this family.
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Whether `format` can back a depth/stencil attachment with OPTIMAL
    /// tiling on this device. Depth/stencil support is device-specific: AMD,
    /// for example, exposes `D32_SFLOAT_S8_UINT` but not `D24_UNORM_S8_UINT`.
    /// The caller must pick a supported format rather than create an image the
    /// driver may silently accept but cannot render (a cascade of validation
    /// errors and a wrong result).
    pub(crate) fn supports_depth_stencil_attachment(&self, format: vk::Format) -> bool {
        // SAFETY: `physical_device` came from this `instance`'s enumeration;
        // the query only fills a properties struct and borrows nothing.
        let props = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device, format)
        };
        props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    }

    pub(crate) fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub(crate) fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// Pick a memory type satisfying `type_bits` (from a `vkGet*MemoryRequirements`)
    /// and containing all of `flags`.
    ///
    /// # Errors
    ///
    /// [`GpuError::VulkanInitFailed`] if the device exposes no such memory type.
    pub(crate) fn find_memory_type(
        &self,
        type_bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<u32, GpuError> {
        let count = self.memory_properties.memory_type_count as usize;
        self.memory_properties.memory_types[..count]
            .iter()
            .enumerate()
            .position(|(i, mt)| (type_bits & (1 << i)) != 0 && mt.property_flags.contains(flags))
            .map(|i| i as u32)
            .ok_or_else(|| {
                GpuError::VulkanInitFailed(format!(
                    "no memory type for bits {type_bits:#x} with {flags:?}"
                ))
            })
    }

    fn create_instance(entry: &Entry, validation: bool) -> Result<(Instance, bool), GpuError> {
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"XPS5X")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"XPS5X")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            // Vulkan 1.3: dynamic rendering is core, so no extension dance.
            .api_version(vk::API_VERSION_1_3);

        let validation_available = validation && Self::has_validation_layer(entry);
        if validation && !validation_available {
            warn!(
                "validation requested but {} is not installed — continuing without it",
                VALIDATION_LAYER.to_string_lossy()
            );
        }

        let mut layers: Vec<*const i8> = Vec::new();
        if validation_available {
            layers.push(VALIDATION_LAYER.as_ptr());
        }

        let mut extensions: Vec<*const i8> = Vec::new();
        if validation_available && Self::has_instance_extension(entry, debug_utils::NAME) {
            extensions.push(debug_utils::NAME.as_ptr());
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        // SAFETY: `create_info` borrows `app_info`, `layers`, and `extensions`,
        // all alive for this call. The layer/extension name pointers are
        // 'static `CStr`s. No allocator callbacks are supplied.
        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
            GpuError::VulkanInitFailed(format!(
                "vkCreateInstance failed (no Vulkan 1.3 driver?): {e}"
            ))
        })?;

        Ok((instance, validation_available))
    }

    fn has_validation_layer(entry: &Entry) -> bool {
        // SAFETY: enumerating layers takes no handles and is valid on any
        // loaded `Entry`. A driver-side failure is treated as "not available".
        let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
            return false;
        };
        layers
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER))
    }

    fn has_instance_extension(entry: &Entry, name: &CStr) -> bool {
        // SAFETY: enumerating instance extensions for the implicit (null)
        // layer takes no handles and is valid on any loaded `Entry`.
        let Ok(exts) = (unsafe { entry.enumerate_instance_extension_properties(None) }) else {
            return false;
        };
        exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
    }

    fn create_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
        let loader = debug_utils::Instance::new(entry, instance);
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        // SAFETY: `loader` was built from a live entry+instance pair, and the
        // callback is a plain `extern "system"` fn with no captured state. The
        // messenger is destroyed in `Drop`/`destroy_partial` before the
        // instance. If the extension is unusable we simply run without it.
        match unsafe { loader.create_debug_utils_messenger(&info, None) } {
            Ok(m) => Some((loader, m)),
            Err(e) => {
                warn!("debug messenger unavailable: {e}");
                None
            }
        }
    }

    /// Select a physical device and create the logical device on it.
    ///
    /// Returns `(physical_device, device, graphics_queue_family, name,
    /// depth_range_unrestricted)`.
    fn pick_and_create_device(
        _entry: &Entry,
        instance: &Instance,
    ) -> Result<(vk::PhysicalDevice, Device, u32, String, bool), GpuError> {
        // SAFETY: `instance` is live; enumeration only fills a Vec of handles.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| GpuError::VulkanInitFailed(format!("device enumeration failed: {e}")))?;

        let mut best: Option<(u32, vk::PhysicalDevice, u32, String)> = None;
        for pd in devices {
            let Some((family, name, score)) = Self::rate_device(instance, pd) else {
                continue;
            };
            debug!("candidate GPU: {name} (score {score})");
            if best.as_ref().is_none_or(|(b, ..)| score > *b) {
                best = Some((score, pd, family, name));
            }
        }

        let (_, physical_device, queue_family_index, device_name) =
            best.ok_or(GpuError::NoSuitableDevice)?;

        // Optional extension: PS5 titles use reverse-Z viewport depth ranges
        // ([1, 0], or outside [0,1]), which core Vulkan forbids. Enable it when
        // the driver offers it; the depth path checks `depth_range_unrestricted`
        // and names the failure when a range cannot be honoured.
        let depth_range_unrestricted =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .map(|exts| {
                    exts.iter().any(|e| {
                        e.extension_name_as_c_str() == Ok(c"VK_EXT_depth_range_unrestricted")
                    })
                })
                .unwrap_or(false);
        let extension_names: Vec<*const i8> = if depth_range_unrestricted {
            vec![c"VK_EXT_depth_range_unrestricted".as_ptr()]
        } else {
            Vec::new()
        };

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];

        // Robustness features when the driver has them: RDNA T#/V# semantics
        // bound-check every buffer and image access in hardware
        // (out-of-bounds reads return 0, writes are dropped), and translated
        // shaders lean on that — a null V# is bound as a 4-byte dummy whose
        // out-of-bounds accesses must be defined, and image/texel-fetch
        // coordinates are whatever the guest computed. Measured on ASTRO.BOT:
        // without robustness one such dispatch is undefined behaviour that
        // takes the whole device down (VK_ERROR_DEVICE_LOST poisons every
        // later draw in the session).
        // SAFETY: `physical_device` is a live handle from this instance and
        // the query structs are local; the calls only write into them.
        let (supported, robust_image_access) = unsafe {
            let mut supported13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut supported2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported13);
            instance.get_physical_device_features2(physical_device, &mut supported2);
            (
                supported2.features,
                supported13.robust_image_access == vk::TRUE,
            )
        };
        // Vulkan 1.3 core features — dynamicRendering is the whole point of
        // targeting 1.3 (no render-pass/framebuffer objects); robustImageAccess
        // is the image half of the RDNA out-of-bounds contract above.
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .robust_image_access(robust_image_access);
        // Gen5 shaders write storage buffers from the vertex stage (UAV
        // writes/stream-out); without `vertexPipelineStoresAndAtomics` the
        // driver rejects those pipelines (VUID-RuntimeSpirv-NonWritable-06341).
        let features = vk::PhysicalDeviceFeatures::default()
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .robust_buffer_access(supported.robust_buffer_access == vk::TRUE);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&features)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features13);

        // SAFETY: `physical_device` came from this `instance`'s enumeration and
        // was verified above to expose `queue_family_index` as a graphics queue
        // and to support `dynamicRendering`. `create_info` borrows locals alive
        // for the call.
        let device = unsafe { instance.create_device(physical_device, &create_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateDevice failed: {e}")))?;

        Ok((
            physical_device,
            device,
            queue_family_index,
            device_name,
            depth_range_unrestricted,
        ))
    }

    /// Score a physical device, or `None` if it cannot run our draw path.
    ///
    /// Requires Vulkan 1.3, `dynamicRendering`, and a graphics queue family.
    /// Discrete GPUs outrank integrated ones.
    fn rate_device(instance: &Instance, pd: vk::PhysicalDevice) -> Option<(u32, String, u32)> {
        // SAFETY: `pd` is a valid handle from this instance; the call fills the
        // properties struct and takes no ownership.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<unnamed GPU>".to_owned());

        if props.api_version < vk::API_VERSION_1_3 {
            debug!("skipping {name}: driver is pre-1.3");
            return None;
        }

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let (fragment_stores_and_atomics, vertex_pipeline_stores_and_atomics) = {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
            // SAFETY: `pd` is valid and `features2` chains `features13`; both
            // live for the call, which only writes into them.
            unsafe { instance.get_physical_device_features2(pd, &mut features2) };
            (
                features2.features.fragment_stores_and_atomics,
                features2.features.vertex_pipeline_stores_and_atomics,
            )
        };
        if features13.dynamic_rendering == vk::FALSE {
            debug!("skipping {name}: no dynamicRendering");
            return None;
        }
        if fragment_stores_and_atomics == vk::FALSE {
            debug!("skipping {name}: no fragmentStoresAndAtomics");
            return None;
        }
        if vertex_pipeline_stores_and_atomics == vk::FALSE {
            // Gen5 vertex shaders can write storage buffers; without the
            // feature those pipelines cannot be created (see device creation).
            debug!("skipping {name}: no vertexPipelineStoresAndAtomics");
            return None;
        }

        // SAFETY: `pd` is valid; the call fills a Vec of queue-family properties.
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))?
            as u32;

        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 250,
            vk::PhysicalDeviceType::CPU => 100,
            _ => 10,
        };
        Some((family, name, score))
    }

    /// Tear down the instance-level handles created before a device existed.
    fn destroy_partial(
        instance: &Instance,
        debug: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    ) {
        // SAFETY: both handles were created from `instance` and are destroyed
        // exactly once here, messenger before instance, with nothing else
        // still referencing them.
        unsafe {
            if let Some((loader, messenger)) = debug {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            instance.destroy_instance(None);
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: every handle below was created by `Self::new` from the
        // matching parent and is destroyed exactly once, children before
        // parents. `device_wait_idle` first guarantees no queued work still
        // references the command pool; its error is ignored because a lost
        // device cannot be recovered here and drop must not panic.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.caches
                .get_mut()
                .destroy(&self.device, self.command_pool);
            if self.pipeline_cache != vk::PipelineCache::null() {
                self.device
                    .destroy_pipeline_cache(self.pipeline_cache, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Routes validation-layer messages into `tracing`.
///
/// # Safety
///
/// Called by the Vulkan loader. `data` must be a valid pointer to a
/// `VkDebugUtilsMessengerCallbackDataEXT` for the duration of the call, which
/// the spec guarantees. We only read `p_message` and never retain it.
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader guarantees `data` is non-null and valid here; we only
    // dereference it for the length of this call.
    let message = unsafe {
        data.as_ref()
            .filter(|d| !d.p_message.is_null())
            .map(|d| CStr::from_ptr(d.p_message).to_string_lossy().into_owned())
            .unwrap_or_default()
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        // Counted as well as logged: see `VALIDATION_ERRORS`. Logging alone is
        // invisible to a consumer with no `tracing` subscriber.
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        tracing::error!("[vulkan] {message}");
    } else {
        warn!("[vulkan] {message}");
    }
    vk::FALSE
}
