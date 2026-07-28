//! Built-in, vendor-neutral reference plugins.
//!
//! These are the plugins Raeen itself ships: all original Rust, no proprietary
//! dependencies, GPL-2.0-clean. They exist to (a) exercise and document the
//! [`PresentPlugin`](super::PresentPlugin) trait, (b) give the present path a
//! working default, and (c) prove the extension point is a *general* upscaler
//! ABI rather than a socket for one proprietary product.
//!
//! Real upscalers (a FidelityFX/FSR pass, which is MIT and can live in-tree)
//! are drop-in replacements for [`NearestUpscale`]; they implement the same
//! trait and read the same [`PresentContext`](super::PresentContext).

use super::{Capabilities, PluginOutput, PresentContext, PresentFrame, PresentPlugin, cabi_v3};
use ash::vk::{self, Handle};
use std::ffi::c_void;

/// Identity: presents the source frame unchanged. Semantically what `active ==
/// None` already does, but registered by name so it can be selected explicitly
/// (e.g. to A/B the plugin path against the fast path, or as the neutral choice
/// in a Settings dropdown).
#[derive(Debug, Default, Clone, Copy)]
pub struct Passthrough;

impl Passthrough {
    pub const NAME: &'static str = "passthrough";
}

impl PresentPlugin for Passthrough {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn process(&mut self, frame: &PresentFrame<'_>, _ctx: &PresentContext) -> PluginOutput {
        PluginOutput::identity(frame)
    }
}

/// Nearest-neighbour spatial upscaler — a minimal but *real* reference that
/// proves the boundary handles a resolution change end-to-end. Scales the frame
/// by [`PresentContext::output_scale`], falling back to identity when no scale
/// is requested or the source is malformed.
///
/// Nearest-neighbour is deliberately the simplest correct resampler: no new
/// dependencies, format-agnostic (it copies whole `bytes_per_pixel` texels, so
/// it works for both the 4-byte display formats and 8-byte HDR). A quality
/// upscaler (FSR) slots in here unchanged from the caller's point of view.
#[derive(Debug, Default, Clone, Copy)]
pub struct NearestUpscale;

impl NearestUpscale {
    pub const NAME: &'static str = "nearest";
}

impl PresentPlugin for NearestUpscale {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            upscale: true,
            ..Default::default()
        }
    }

    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        let scale = ctx.output_scale.clamp(1.0, 8.0);
        let bpp = frame.bytes_per_pixel as usize;
        let dst_w = ((frame.width as f32 * scale).round() as u32).max(1);
        let dst_h = ((frame.height as f32 * scale).round() as u32).max(1);

        let src_texels = (frame.width as usize).saturating_mul(frame.height as usize);
        let src_ok = bpp != 0 && frame.color.len() >= src_texels.saturating_mul(bpp);

        // Nothing to do (or can't safely do it) → identity.
        if (dst_w == frame.width && dst_h == frame.height) || !src_ok {
            return PluginOutput::identity(frame);
        }

        let mut pixels = vec![0u8; dst_w as usize * dst_h as usize * bpp];
        for y in 0..dst_h {
            // Map destination row to nearest source row.
            let sy = ((y as u64 * frame.height as u64) / dst_h as u64).min(frame.height as u64 - 1)
                as u32;
            for x in 0..dst_w {
                let sx = ((x as u64 * frame.width as u64) / dst_w as u64)
                    .min(frame.width as u64 - 1) as u32;
                let src = ((sy * frame.width + sx) as usize) * bpp;
                let dst = ((y * dst_w + x) as usize) * bpp;
                pixels[dst..dst + bpp].copy_from_slice(&frame.color[src..src + bpp]);
            }
        }

        PluginOutput {
            primary: super::PluginFrame {
                width: dst_w,
                height: dst_h,
                bytes_per_pixel: frame.bytes_per_pixel,
                pixels,
            },
            generated: Vec::new(),
        }
    }
}

/// Vulkan linear-blit upscaler used to exercise ABI v3 end to end.
///
/// It owns no Vulkan objects and records one blit into the host command buffer.
/// The host owns resource transitions, submission, synchronization, and
/// readback. This is intentionally simple: its purpose is to prove that the
/// generic GPU plugin path works without any vendor SDK.
#[derive(Default)]
pub struct VulkanBlitUpscale {
    device: Option<ash::Device>,
}

impl VulkanBlitUpscale {
    pub const NAME: &'static str = "gpu-blit";
}

impl PresentPlugin for VulkanBlitUpscale {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            upscale: true,
            gpu_frames: true,
            ..Default::default()
        }
    }

    fn vulkan_requirements_v3(&self) -> Option<cabi_v3::RaeenVulkanRequirementsV3> {
        let mut requirements = cabi_v3::RaeenVulkanRequirementsV3::empty();
        requirements.minimum_api_version = vk::API_VERSION_1_3;
        requirements.required_queue_flags = cabi_v3::RAEEN_V3_QUEUE_GRAPHICS;
        requirements.required_feature_flags = cabi_v3::RAEEN_V3_FEATURE_TIMELINE_SEMAPHORE;
        Some(requirements)
    }

    fn initialize_gpu_v3(&mut self, host: &cabi_v3::RaeenVulkanHostV3) -> Result<(), String> {
        if host.device == 0 || host.get_device_proc_addr.is_null() {
            return Err("gpu-blit received an incomplete Vulkan host".to_owned());
        }
        // SAFETY: ABI v3 guarantees this pointer is vkGetDeviceProcAddr for
        // `host.device` and remains valid until shutdown_gpu_v3.
        let get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr =
            unsafe { std::mem::transmute(host.get_device_proc_addr) };
        let raw_device = vk::Device::from_raw(host.device);
        // SAFETY: every function is resolved from the live host device. The
        // wrapper does not own/destroy the VkDevice.
        self.device = Some(unsafe {
            ash::Device::load_with(
                |name| {
                    get_device_proc_addr(raw_device, name.as_ptr())
                        .map_or(std::ptr::null(), |function| function as *const c_void)
                },
                raw_device,
            )
        });
        Ok(())
    }

    fn shutdown_gpu_v3(&mut self) {
        self.device = None;
    }

    fn process_gpu_v3(
        &mut self,
        frame: &cabi_v3::RaeenPresentFrameV3,
        output: &mut cabi_v3::RaeenPluginOutputV3,
    ) -> i32 {
        let Some(device) = &self.device else {
            return cabi_v3::RAEEN_V3_DECLINED;
        };
        let command_buffer = vk::CommandBuffer::from_raw(frame.command_buffer);
        let color = vk::Image::from_raw(frame.color.image);
        let destination = vk::Image::from_raw(frame.output.image);
        let color_layout = vk::ImageLayout::from_raw(frame.color.layout as i32);
        let output_layout = vk::ImageLayout::from_raw(frame.output.layout as i32);
        let blit = vk::ImageBlit::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offsets([
                vk::Offset3D {
                    x: frame.render_rect.x as i32,
                    y: frame.render_rect.y as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: (frame.render_rect.x + frame.render_rect.width) as i32,
                    y: (frame.render_rect.y + frame.render_rect.height) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D {
                    x: frame.output_rect.x as i32,
                    y: frame.output_rect.y as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: (frame.output_rect.x + frame.output_rect.width) as i32,
                    y: (frame.output_rect.y + frame.output_rect.height) as i32,
                    z: 1,
                },
            ]);
        // SAFETY: the host validates and owns both resources, has transitioned
        // them to the declared layouts, and keeps them alive through submit.
        unsafe {
            device.cmd_blit_image(
                command_buffer,
                color,
                color_layout,
                destination,
                output_layout,
                &[blit],
                vk::Filter::LINEAR,
            );
        }
        output.output_layout = frame.output.layout;
        cabi_v3::RAEEN_V3_OK
    }

    fn process(&mut self, frame: &PresentFrame<'_>, _ctx: &PresentContext) -> PluginOutput {
        // GPU work already occurred before readback. CPU-only callers retain a
        // safe identity fallback.
        PluginOutput::identity(frame)
    }
}
