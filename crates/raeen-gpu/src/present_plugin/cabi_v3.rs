//! Present-plugin ABI v3: GPU resources, temporal metadata, and pre-device
//! Vulkan requirements.
//!
//! This module is vendor-neutral and contains no upscaler implementation. It
//! defines the host/plugin boundary needed by temporal Vulkan passes such as
//! FSR2/3, XeSS, or a separately distributed DLSS adapter. In particular:
//!
//! - requirements are queried before the logical device is created;
//! - every image is host-owned and only borrowed by the plugin;
//! - the host supplies a recording command buffer and synchronization points;
//! - depth, motion, exposure, jitter, history reset, and subrects are explicit;
//! - a plugin records work into a host-owned output image instead of returning
//!   an allocation with ambiguous lifetime.

use std::ffi::c_void;

pub const RAEEN_PLUGIN_ABI_VERSION_V3: u32 = 3;
pub const RAEEN_PLUGIN_ENTRY_V3: &[u8] = b"raeen_plugin_v3\0";

pub const RAEEN_V3_OK: i32 = 0;
pub const RAEEN_V3_DECLINED: i32 = 1;
pub const RAEEN_V3_BAD_INPUT: i32 = -1;

pub const RAEEN_V3_MAX_INSTANCE_EXTENSIONS: usize = 16;
pub const RAEEN_V3_MAX_DEVICE_EXTENSIONS: usize = 32;
pub const RAEEN_V3_MAX_EXTENSION_NAME: usize = 256;

pub const RAEEN_V3_QUEUE_GRAPHICS: u32 = 1 << 0;
pub const RAEEN_V3_QUEUE_COMPUTE: u32 = 1 << 1;
pub const RAEEN_V3_QUEUE_OPTICAL_FLOW: u32 = 1 << 2;

pub const RAEEN_V3_FEATURE_TIMELINE_SEMAPHORE: u64 = 1 << 0;
pub const RAEEN_V3_FEATURE_DESCRIPTOR_INDEXING: u64 = 1 << 1;
pub const RAEEN_V3_FEATURE_BUFFER_DEVICE_ADDRESS: u64 = 1 << 2;
pub const RAEEN_V3_KNOWN_FEATURES: u64 = RAEEN_V3_FEATURE_TIMELINE_SEMAPHORE
    | RAEEN_V3_FEATURE_DESCRIPTOR_INDEXING
    | RAEEN_V3_FEATURE_BUFFER_DEVICE_ADDRESS;

pub const RAEEN_V3_RESOURCE_BORROWED: u32 = 1 << 0;
pub const RAEEN_V3_RESOURCE_HOST_OWNS_LAYOUT: u32 = 1 << 1;

pub const RAEEN_V3_TEMPORAL_RESET: u32 = 1 << 0;
pub const RAEEN_V3_DEPTH_INVERTED: u32 = 1 << 1;
pub const RAEEN_V3_DEPTH_INFINITE: u32 = 1 << 2;
pub const RAEEN_V3_MOTION_VECTORS_DILATED: u32 = 1 << 3;
pub const RAEEN_V3_MOTION_VECTORS_JITTERED: u32 = 1 << 4;
pub const RAEEN_V3_ORTHOGRAPHIC: u32 = 1 << 5;
pub const RAEEN_V3_HAS_EXPOSURE_TEXTURE: u32 = 1 << 6;

/// One length-bounded Vulkan extension name. The bytes are UTF-8 without a NUL.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaeenExtensionNameV3 {
    pub len: u32,
    pub bytes: [u8; RAEEN_V3_MAX_EXTENSION_NAME],
}

impl RaeenExtensionNameV3 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            len: 0,
            bytes: [0; RAEEN_V3_MAX_EXTENSION_NAME],
        }
    }

    pub fn as_str(&self) -> Result<&str, V3ValidationError> {
        let len = self.len as usize;
        if len == 0 || len >= self.bytes.len() {
            return Err(V3ValidationError::BadExtensionName);
        }
        let bytes = &self.bytes[..len];
        if bytes.contains(&0) {
            return Err(V3ValidationError::BadExtensionName);
        }
        std::str::from_utf8(bytes).map_err(|_| V3ValidationError::BadExtensionName)
    }
}

/// Requirements returned before `vkCreateInstance`/`vkCreateDevice`.
///
/// Fixed-capacity arrays keep all memory host-owned: an untrusted plugin never
/// hands Raeen a pointer/count pair to walk during device initialization.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenVulkanRequirementsV3 {
    pub struct_size: u32,
    pub minimum_api_version: u32,
    pub instance_extension_count: u32,
    pub device_extension_count: u32,
    pub instance_extensions: [RaeenExtensionNameV3; RAEEN_V3_MAX_INSTANCE_EXTENSIONS],
    pub device_extensions: [RaeenExtensionNameV3; RAEEN_V3_MAX_DEVICE_EXTENSIONS],
    pub required_queue_flags: u32,
    pub extra_graphics_queues: u32,
    pub extra_compute_queues: u32,
    pub extra_optical_flow_queues: u32,
    /// Reserved, vendor-neutral feature bits. A host must reject unknown
    /// required bits rather than silently creating an insufficient device.
    pub required_feature_flags: u64,
    pub _reserved: [u64; 7],
}

impl RaeenVulkanRequirementsV3 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            minimum_api_version: 0,
            instance_extension_count: 0,
            device_extension_count: 0,
            instance_extensions: [RaeenExtensionNameV3::empty(); RAEEN_V3_MAX_INSTANCE_EXTENSIONS],
            device_extensions: [RaeenExtensionNameV3::empty(); RAEEN_V3_MAX_DEVICE_EXTENSIONS],
            required_queue_flags: 0,
            extra_graphics_queues: 0,
            extra_compute_queues: 0,
            extra_optical_flow_queues: 0,
            required_feature_flags: 0,
            _reserved: [0; 7],
        }
    }

    pub fn validate(&self) -> Result<(), V3ValidationError> {
        if self.struct_size as usize != std::mem::size_of::<Self>() {
            return Err(V3ValidationError::BadStructSize);
        }
        let instance_count = self.instance_extension_count as usize;
        let device_count = self.device_extension_count as usize;
        if instance_count > RAEEN_V3_MAX_INSTANCE_EXTENSIONS
            || device_count > RAEEN_V3_MAX_DEVICE_EXTENSIONS
        {
            return Err(V3ValidationError::TooManyExtensions);
        }
        for name in &self.instance_extensions[..instance_count] {
            name.as_str()?;
        }
        for name in &self.device_extensions[..device_count] {
            name.as_str()?;
        }
        let known_queues =
            RAEEN_V3_QUEUE_GRAPHICS | RAEEN_V3_QUEUE_COMPUTE | RAEEN_V3_QUEUE_OPTICAL_FLOW;
        if self.required_queue_flags & !known_queues != 0
            || self.required_feature_flags & !RAEEN_V3_KNOWN_FEATURES != 0
        {
            return Err(V3ValidationError::UnknownRequiredCapability);
        }
        Ok(())
    }
}

/// Vulkan dispatch and queues provided after requirements have been enabled.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenVulkanHostV3 {
    pub instance: u64,
    pub physical_device: u64,
    pub device: u64,
    pub graphics_queue: u64,
    pub compute_queue: u64,
    pub optical_flow_queue: u64,
    pub graphics_queue_family: u32,
    pub compute_queue_family: u32,
    pub optical_flow_queue_family: u32,
    pub _reserved: u32,
    pub get_instance_proc_addr: *const c_void,
    pub get_device_proc_addr: *const c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaeenRectV3 {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A borrowed Vulkan image. The plugin must not destroy the image, view, or
/// memory, and may access it only while `process` is executing.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenVulkanResourceV3 {
    pub image: u64,
    pub image_view: u64,
    pub device_memory: u64,
    pub vk_format: u32,
    pub layout: u32,
    pub width: u32,
    pub height: u32,
    pub queue_family: u32,
    pub flags: u32,
}

impl RaeenVulkanResourceV3 {
    #[must_use]
    pub const fn absent() -> Self {
        Self {
            image: 0,
            image_view: 0,
            device_memory: 0,
            vk_format: 0,
            layout: 0,
            width: 0,
            height: 0,
            queue_family: 0,
            flags: 0,
        }
    }

    #[must_use]
    pub const fn is_present(&self) -> bool {
        self.image != 0
    }
}

/// Per-frame timeline-semaphore contract.
///
/// The host waits on `signal_semaphore/signal_value` before sampling output.
/// The plugin records into `command_buffer`; it does not submit the queue.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenFrameSyncV3 {
    pub wait_semaphore: u64,
    pub wait_value: u64,
    pub signal_semaphore: u64,
    pub signal_value: u64,
}

/// Temporal constants use Vulkan clip-space conventions. Matrices are
/// row-major. Motion-vector scale converts stored motion texels into pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenTemporalDataV3 {
    pub flags: u32,
    pub _reserved: u32,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub motion_vector_scale_x: f32,
    pub motion_vector_scale_y: f32,
    pub exposure_scale: f32,
    pub pre_exposure: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub frame_time_ms: f32,
    pub camera_view_to_clip: [f32; 16],
    pub camera_clip_to_view: [f32; 16],
    pub camera_clip_to_previous_clip: [f32; 16],
    pub camera_previous_clip_to_clip: [f32; 16],
}

/// Complete GPU-resident input for one temporal present pass.
///
/// `output` is allocated by Raeen and starts in the declared layout. The
/// command buffer is recording, outside a render pass, and submitted by Raeen
/// after `process` returns successfully.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPresentFrameV3 {
    pub struct_size: u32,
    pub _reserved: u32,
    pub frame_index: u64,
    pub command_buffer: u64,
    pub color: RaeenVulkanResourceV3,
    pub depth: RaeenVulkanResourceV3,
    pub motion_vectors: RaeenVulkanResourceV3,
    pub exposure: RaeenVulkanResourceV3,
    pub output: RaeenVulkanResourceV3,
    pub render_rect: RaeenRectV3,
    pub output_rect: RaeenRectV3,
    pub temporal: RaeenTemporalDataV3,
    pub sync: RaeenFrameSyncV3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginOutputV3 {
    pub struct_size: u32,
    /// Final `VkImageLayout` of the host-owned output image.
    pub output_layout: u32,
    pub _reserved: [u64; 4],
}

impl RaeenPluginOutputV3 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            output_layout: 0,
            _reserved: [0; 4],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V3ValidationError {
    BadStructSize,
    BadExtensionName,
    TooManyExtensions,
    UnknownRequiredCapability,
    MissingCommandBuffer,
    MissingResource(&'static str),
    BadResourceFlags(&'static str),
    BadRect(&'static str),
    BadTemporalValue(&'static str),
    BadSynchronization,
}

fn validate_required_resource(
    resource: &RaeenVulkanResourceV3,
    name: &'static str,
) -> Result<(), V3ValidationError> {
    if resource.image == 0
        || resource.image_view == 0
        || resource.width == 0
        || resource.height == 0
        || resource.vk_format == 0
    {
        return Err(V3ValidationError::MissingResource(name));
    }
    let required_flags = RAEEN_V3_RESOURCE_BORROWED | RAEEN_V3_RESOURCE_HOST_OWNS_LAYOUT;
    if resource.flags & required_flags != required_flags {
        return Err(V3ValidationError::BadResourceFlags(name));
    }
    Ok(())
}

fn validate_rect(
    rect: RaeenRectV3,
    resource: &RaeenVulkanResourceV3,
    name: &'static str,
) -> Result<(), V3ValidationError> {
    let x2 = rect
        .x
        .checked_add(rect.width)
        .ok_or(V3ValidationError::BadRect(name))?;
    let y2 = rect
        .y
        .checked_add(rect.height)
        .ok_or(V3ValidationError::BadRect(name))?;
    if rect.width == 0 || rect.height == 0 || x2 > resource.width || y2 > resource.height {
        return Err(V3ValidationError::BadRect(name));
    }
    Ok(())
}

impl RaeenPresentFrameV3 {
    pub fn validate(&self) -> Result<(), V3ValidationError> {
        self.validate_for_inputs(true, true)
    }

    /// Validate the common GPU-frame contract and only the auxiliary inputs
    /// advertised by the active plugin.
    pub fn validate_for_inputs(
        &self,
        wants_depth: bool,
        wants_motion_vectors: bool,
    ) -> Result<(), V3ValidationError> {
        if self.struct_size as usize != std::mem::size_of::<Self>() {
            return Err(V3ValidationError::BadStructSize);
        }
        if self.command_buffer == 0 {
            return Err(V3ValidationError::MissingCommandBuffer);
        }
        validate_required_resource(&self.color, "color")?;
        if wants_depth {
            validate_required_resource(&self.depth, "depth")?;
        }
        if wants_motion_vectors {
            validate_required_resource(&self.motion_vectors, "motion_vectors")?;
        }
        validate_required_resource(&self.output, "output")?;
        validate_rect(self.render_rect, &self.color, "render_rect")?;
        validate_rect(self.output_rect, &self.output, "output_rect")?;

        if self.temporal.flags & RAEEN_V3_HAS_EXPOSURE_TEXTURE != 0 {
            validate_required_resource(&self.exposure, "exposure")?;
        }
        for (name, value) in [
            ("jitter_x", self.temporal.jitter_x),
            ("jitter_y", self.temporal.jitter_y),
            ("motion_vector_scale_x", self.temporal.motion_vector_scale_x),
            ("motion_vector_scale_y", self.temporal.motion_vector_scale_y),
            ("exposure_scale", self.temporal.exposure_scale),
            ("pre_exposure", self.temporal.pre_exposure),
            ("near_plane", self.temporal.near_plane),
            ("far_plane", self.temporal.far_plane),
            ("frame_time_ms", self.temporal.frame_time_ms),
        ] {
            if !value.is_finite() {
                return Err(V3ValidationError::BadTemporalValue(name));
            }
        }
        if self.temporal.motion_vector_scale_x == 0.0
            || self.temporal.motion_vector_scale_y == 0.0
            || self.temporal.frame_time_ms < 0.0
        {
            return Err(V3ValidationError::BadTemporalValue(
                "motion scale/frame time",
            ));
        }
        if self.sync.wait_semaphore == 0
            || self.sync.signal_semaphore == 0
            || self.sync.signal_value <= self.sync.wait_value
        {
            return Err(V3ValidationError::BadSynchronization);
        }
        Ok(())
    }
}

pub type RaeenQueryRequirementsV3Fn = unsafe extern "C" fn(*mut RaeenVulkanRequirementsV3) -> i32;
pub type RaeenCreateV3Fn = unsafe extern "C" fn(*const RaeenVulkanHostV3) -> *mut c_void;
pub type RaeenDestroyV3Fn = unsafe extern "C" fn(*mut c_void);
pub type RaeenNameV3Fn = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> usize;
pub type RaeenCapabilitiesV3Fn = unsafe extern "C" fn(*mut c_void) -> u32;
pub type RaeenProcessV3Fn =
    unsafe extern "C" fn(*mut c_void, *const RaeenPresentFrameV3, *mut RaeenPluginOutputV3) -> i32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginV3 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub query_requirements: RaeenQueryRequirementsV3Fn,
    pub create: RaeenCreateV3Fn,
    pub destroy: RaeenDestroyV3Fn,
    pub name: RaeenNameV3Fn,
    pub capabilities: RaeenCapabilitiesV3Fn,
    pub process: RaeenProcessV3Fn,
    pub _reserved: [usize; 8],
}

pub type RaeenPluginEntryV3Fn = unsafe extern "C" fn() -> *const RaeenPluginV3;

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(width: u32, height: u32) -> RaeenVulkanResourceV3 {
        RaeenVulkanResourceV3 {
            image: 1,
            image_view: 2,
            device_memory: 3,
            vk_format: 37,
            layout: 5,
            width,
            height,
            queue_family: 0,
            flags: RAEEN_V3_RESOURCE_BORROWED | RAEEN_V3_RESOURCE_HOST_OWNS_LAYOUT,
        }
    }

    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn valid_frame() -> RaeenPresentFrameV3 {
        RaeenPresentFrameV3 {
            struct_size: std::mem::size_of::<RaeenPresentFrameV3>() as u32,
            _reserved: 0,
            frame_index: 9,
            command_buffer: 4,
            color: resource(1280, 720),
            depth: resource(1280, 720),
            motion_vectors: resource(1280, 720),
            exposure: RaeenVulkanResourceV3::absent(),
            output: resource(1920, 1080),
            render_rect: RaeenRectV3 {
                x: 0,
                y: 0,
                width: 1280,
                height: 720,
            },
            output_rect: RaeenRectV3 {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            },
            temporal: RaeenTemporalDataV3 {
                flags: RAEEN_V3_TEMPORAL_RESET,
                _reserved: 0,
                jitter_x: 0.25,
                jitter_y: -0.25,
                motion_vector_scale_x: 1280.0,
                motion_vector_scale_y: 720.0,
                exposure_scale: 1.0,
                pre_exposure: 1.0,
                near_plane: 0.1,
                far_plane: 1000.0,
                frame_time_ms: 16.6,
                camera_view_to_clip: identity(),
                camera_clip_to_view: identity(),
                camera_clip_to_previous_clip: identity(),
                camera_previous_clip_to_clip: identity(),
            },
            sync: RaeenFrameSyncV3 {
                wait_semaphore: 5,
                wait_value: 11,
                signal_semaphore: 6,
                signal_value: 12,
            },
        }
    }

    #[test]
    fn complete_temporal_frame_validates() {
        assert_eq!(valid_frame().validate(), Ok(()));
    }

    #[test]
    fn subrect_must_fit_its_resource() {
        let mut frame = valid_frame();
        frame.render_rect.x = 1200;
        assert_eq!(
            frame.validate(),
            Err(V3ValidationError::BadRect("render_rect"))
        );
    }

    #[test]
    fn temporal_values_must_be_finite_and_motion_scale_nonzero() {
        let mut frame = valid_frame();
        frame.temporal.jitter_x = f32::NAN;
        assert_eq!(
            frame.validate(),
            Err(V3ValidationError::BadTemporalValue("jitter_x"))
        );
        frame = valid_frame();
        frame.temporal.motion_vector_scale_x = 0.0;
        assert_eq!(
            frame.validate(),
            Err(V3ValidationError::BadTemporalValue(
                "motion scale/frame time"
            ))
        );
    }

    #[test]
    fn exposure_texture_is_required_only_when_flagged() {
        let mut frame = valid_frame();
        frame.temporal.flags |= RAEEN_V3_HAS_EXPOSURE_TEXTURE;
        assert_eq!(
            frame.validate(),
            Err(V3ValidationError::MissingResource("exposure"))
        );
        frame.exposure = resource(1, 1);
        assert_eq!(frame.validate(), Ok(()));
    }

    #[test]
    fn synchronization_must_advance() {
        let mut frame = valid_frame();
        frame.sync.signal_value = frame.sync.wait_value;
        assert_eq!(frame.validate(), Err(V3ValidationError::BadSynchronization));
    }

    #[test]
    fn requirement_names_are_bounded_and_nul_free() {
        let mut requirements = RaeenVulkanRequirementsV3::empty();
        requirements.instance_extension_count = 1;
        let name = b"VK_KHR_external_memory_capabilities";
        requirements.instance_extensions[0].len = name.len() as u32;
        requirements.instance_extensions[0].bytes[..name.len()].copy_from_slice(name);
        assert_eq!(requirements.validate(), Ok(()));

        requirements.instance_extensions[0].bytes[2] = 0;
        assert_eq!(
            requirements.validate(),
            Err(V3ValidationError::BadExtensionName)
        );
    }

    #[test]
    fn known_vulkan_feature_requirements_are_explicit() {
        let mut requirements = RaeenVulkanRequirementsV3::empty();
        requirements.required_feature_flags = RAEEN_V3_FEATURE_TIMELINE_SEMAPHORE
            | RAEEN_V3_FEATURE_DESCRIPTOR_INDEXING
            | RAEEN_V3_FEATURE_BUFFER_DEVICE_ADDRESS;
        assert_eq!(requirements.validate(), Ok(()));

        requirements.required_feature_flags |= 1 << 63;
        assert_eq!(
            requirements.validate(),
            Err(V3ValidationError::UnknownRequiredCapability)
        );
    }
}
