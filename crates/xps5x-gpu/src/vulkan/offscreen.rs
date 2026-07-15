//! Offscreen triangle draw with pixel readback.
//!
//! Renders one triangle into a device-local `R8G8B8A8_UNORM` image using
//! Vulkan 1.3 dynamic rendering (no render-pass/framebuffer objects), copies
//! the image into a host-visible buffer, and hands the pixels back to the
//! caller.
//!
//! This exists so the GPU draw path is verifiable **headlessly**: no window, no
//! swapchain, no surface extensions. A test can assert on actual rasterized
//! pixels, which is the only honest proof that the pipeline drew anything.
//! Presentation to a real swapchain is a separate, later concern.

use super::instance::VulkanDevice;
use super::shaders::{triangle_fragment_spirv, triangle_vertex_spirv};
use ash::{Device, util::Align, vk};
use std::mem;
use tracing::info;
use xps5x_core::error::GpuError;

/// The color the attachment is cleared to before the draw, as linear RGBA.
///
/// Deliberately not black and not fully-saturated: a readback buffer that was
/// never written (all zeroes) or a mis-sized copy cannot masquerade as a
/// correct clear.
pub const CLEAR_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// Bytes per pixel of the render target's `R8G8B8A8_UNORM` format.
const BYTES_PER_PIXEL: u32 = 4;

/// Triangle vertices in Vulkan normalized device coordinates (`x, y, z, w`).
///
/// Vulkan's NDC has +Y pointing **down**, so the `-0.7` vertex is the top one.
/// The triangle is sized to cover the image center while leaving every corner
/// outside it — that gap is exactly what the acceptance test asserts on.
const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0], // top
    [0.7, 0.7, 0.0, 1.0],  // bottom right
    [-0.7, 0.7, 0.0, 1.0], // bottom left
];

/// A rendered image read back from the GPU: tightly-packed RGBA8 rows.
#[derive(Debug, Clone)]
pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, row-major, no padding.
    pub pixels: Vec<u8>,
}

impl RenderedImage {
    /// The RGBA bytes at `(x, y)`, or `None` if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = ((y * self.width + x) * BYTES_PER_PIXEL) as usize;
        self.pixels
            .get(offset..offset + 4)
            .map(|p| [p[0], p[1], p[2], p[3]])
    }
}

/// Convert a linear float color to the `R8G8B8A8_UNORM` bytes it should
/// produce, for comparing against readback.
pub fn unorm8(color: [f32; 4]) -> [u8; 4] {
    color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Draw one triangle offscreen at `width` x `height` and read back the pixels.
///
/// # Errors
///
/// [`GpuError::VulkanInitFailed`] if any resource creation or submission fails,
/// [`GpuError::PipelineCreationFailed`] if the graphics pipeline is rejected.
pub fn render_triangle(
    dev: &VulkanDevice,
    width: u32,
    height: u32,
) -> Result<RenderedImage, GpuError> {
    if width == 0 || height == 0 {
        return Err(GpuError::VulkanInitFailed(format!(
            "invalid render target size {width}x{height}"
        )));
    }

    let mut res = Resources::new(dev);
    res.build(width, height)?;
    res.record_and_submit(width, height)?;
    let pixels = res.read_back(width, height)?;

    info!(
        "offscreen triangle rendered at {width}x{height} on {}",
        dev.device_name()
    );
    Ok(RenderedImage {
        width,
        height,
        pixels,
    })
}

/// Owns every Vulkan handle the draw needs.
///
/// Handles start null and are filled in by `build`. `Drop` destroys whatever is
/// non-null, so an error at any step during `build` cleans up correctly rather
/// than leaking GPU memory — `?` early-returns are safe here.
struct Resources<'a> {
    dev: &'a VulkanDevice,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl<'a> Resources<'a> {
    fn new(dev: &'a VulkanDevice) -> Self {
        Self {
            dev,
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            image_view: vk::ImageView::null(),
            vertex_buffer: vk::Buffer::null(),
            vertex_memory: vk::DeviceMemory::null(),
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            vertex_module: vk::ShaderModule::null(),
            fragment_module: vk::ShaderModule::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        }
    }

    fn device(&self) -> &Device {
        self.dev.device()
    }

    fn build(&mut self, width: u32, height: u32) -> Result<(), GpuError> {
        self.create_render_target(width, height)?;
        self.create_vertex_buffer()?;
        self.create_readback_buffer(width, height)?;
        self.create_pipeline()?;
        self.create_command_resources()?;
        Ok(())
    }

    fn create_render_target(&mut self, width: u32, height: u32) -> Result<(), GpuError> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // COLOR_ATTACHMENT to draw into it, TRANSFER_SRC to copy it out.
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live. The handle is stored and destroyed in Drop.
        self.image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateImage failed: {e}")))?;

        // SAFETY: `self.image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(self.image) };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);

        // SAFETY: allocation size/type come from this image's own requirements.
        self.image_memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("image allocation failed: {e}")))?;

        // SAFETY: memory was allocated for exactly this image, offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe {
            self.device()
                .bind_image_memory(self.image, self.image_memory, 0)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkBindImageMemory failed: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        self.image_view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateImageView failed: {e}")))?;
        Ok(())
    }

    /// Create a buffer plus memory satisfying `properties`.
    fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: `info` is fully initialized; the device is live.
        let buffer = unsafe { self.device().create_buffer(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateBuffer failed: {e}")))?;

        // SAFETY: `buffer` was just created from this device.
        let reqs = unsafe { self.device().get_buffer_memory_requirements(buffer) };
        let type_index = match self.dev.find_memory_type(reqs.memory_type_bits, properties) {
            Ok(i) => i,
            Err(e) => {
                // SAFETY: destroying the buffer we just created and are about
                // to drop the only handle to; nothing references it yet.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(e);
            }
        };

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);

        // SAFETY: size/type come from this buffer's own requirements.
        let memory = match unsafe { self.device().allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: same as above — unbound buffer, sole handle.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(GpuError::VulkanInitFailed(format!(
                    "buffer allocation failed: {e}"
                )));
            }
        };

        // SAFETY: memory was allocated for exactly this buffer; offset 0 is in
        // range and correctly aligned by construction.
        if let Err(e) = unsafe { self.device().bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: unwinding our own two handles, neither yet in use.
            unsafe {
                self.device().free_memory(memory, None);
                self.device().destroy_buffer(buffer, None);
            }
            return Err(GpuError::VulkanInitFailed(format!(
                "vkBindBufferMemory failed: {e}"
            )));
        }
        Ok((buffer, memory))
    }

    fn create_vertex_buffer(&mut self) -> Result<(), GpuError> {
        let size = mem::size_of_val(&TRIANGLE_VERTICES) as vk::DeviceSize;
        let (buffer, memory) = self.create_buffer(
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            // HOST_COHERENT so the write is visible to the GPU without an
            // explicit flush.
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.vertex_buffer = buffer;
        self.vertex_memory = memory;

        // SAFETY: the memory is HOST_VISIBLE, not currently mapped, and the
        // whole allocation is requested. The GPU is not using it yet — nothing
        // has been submitted.
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory failed: {e}")))?;

        // SAFETY: `ptr` maps `size` bytes allocated for this buffer, and
        // `Align` writes exactly the slice we pass within that range. The
        // alignment of `[f32; 4]` is 4, well under the mapped base alignment
        // Vulkan guarantees.
        unsafe {
            let mut align = Align::new(ptr, mem::align_of::<[f32; 4]>() as u64, size);
            align.copy_from_slice(&TRIANGLE_VERTICES);
            self.device().unmap_memory(memory);
        }
        Ok(())
    }

    fn create_readback_buffer(&mut self, width: u32, height: u32) -> Result<(), GpuError> {
        let size = vk::DeviceSize::from(width)
            * vk::DeviceSize::from(height)
            * vk::DeviceSize::from(BYTES_PER_PIXEL);
        let (buffer, memory) = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.readback_buffer = buffer;
        self.readback_memory = memory;
        Ok(())
    }

    fn create_shader_module(&self, code: &[u32]) -> Result<vk::ShaderModule, GpuError> {
        let info = vk::ShaderModuleCreateInfo::default().code(code);
        // SAFETY: `code` is a `&[u32]`, so it is 4-byte aligned and its length
        // is a whole number of words — exactly what vkCreateShaderModule
        // requires. It stays alive for the call.
        unsafe { self.device().create_shader_module(&info, None) }
            .map_err(|e| GpuError::ShaderCompilationFailed(format!("vkCreateShaderModule: {e}")))
    }

    fn create_pipeline(&mut self) -> Result<(), GpuError> {
        self.vertex_module = self.create_shader_module(&triangle_vertex_spirv())?;
        self.fragment_module = self.create_shader_module(&triangle_fragment_spirv())?;

        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: an empty layout — no descriptor sets or push constants.
        self.pipeline_layout = unsafe { self.device().create_pipeline_layout(&layout_info, None) }
            .map_err(|e| {
                GpuError::PipelineCreationFailed(format!("vkCreatePipelineLayout: {e}"))
            })?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment_module)
                .name(c"main"),
        ];

        // One vec4 attribute at location 0, matching the vertex shader.
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<[f32; 4]>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(0)];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Viewport and scissor are dynamic, set during recording.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            // No culling: the test asserts on coverage, not winding order, and
            // this keeps the result independent of vertex order.
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Vulkan 1.3 dynamic rendering: the pipeline declares the attachment
        // formats directly instead of referencing a VkRenderPass.
        let color_formats = [vk::Format::R8G8B8A8_UNORM];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_info);

        // SAFETY: every struct chained into `pipeline_info` is a local alive
        // for this call; the shader modules and layout are live handles from
        // this device. A null pipeline cache is valid.
        let pipelines = unsafe {
            self.device().create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| {
            GpuError::PipelineCreationFailed(format!("vkCreateGraphicsPipelines: {e}"))
        })?;

        self.pipeline = *pipelines.first().ok_or_else(|| {
            GpuError::PipelineCreationFailed("driver returned no pipeline".to_owned())
        })?;
        Ok(())
    }

    fn create_command_resources(&mut self) -> Result<(), GpuError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.dev.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: the pool belongs to this device and is not being used
        // concurrently — `VulkanDevice` is borrowed immutably and command
        // buffers are only recorded here, on this thread.
        let buffers = unsafe { self.device().allocate_command_buffers(&alloc_info) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("command buffer alloc: {e}")))?;
        self.command_buffer = *buffers
            .first()
            .ok_or_else(|| GpuError::VulkanInitFailed("no command buffer returned".to_owned()))?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: plain unsignaled fence on a live device.
        self.fence = unsafe { self.device().create_fence(&fence_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateFence failed: {e}")))?;
        Ok(())
    }

    /// Barrier helper: transition the render target between layouts.
    fn image_barrier(
        &self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // SAFETY: called only between begin/end of `self.command_buffer`, which
        // is in the recording state. The barrier names this struct's own live
        // image and a subresource range within its creation parameters.
        unsafe {
            self.device().cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    fn record_and_submit(&mut self, width: u32, height: u32) -> Result<(), GpuError> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: the command buffer is freshly allocated and not pending, so
        // beginning it is legal; it is recorded only from this thread.
        unsafe {
            self.device()
                .begin_command_buffer(self.command_buffer, &begin_info)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkBeginCommandBuffer: {e}")))?;

        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL. Discards existing contents,
        // which is fine: the render pass clears anyway.
        self.image_barrier(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: CLEAR_COLOR,
            },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let color_attachments = [color_attachment];

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [render_area];

        // SAFETY: all handles below belong to this device and are live; the
        // command buffer is recording; the vertex buffer holds 3 vertices,
        // matching the `draw(3, ...)` call; the pipeline's attachment format
        // matches the image view's. `cmd_begin_rendering` is core in Vulkan
        // 1.3 and `dynamicRendering` was required at device selection.
        unsafe {
            let d = self.device();
            d.cmd_begin_rendering(self.command_buffer, &rendering_info);
            d.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            d.cmd_set_viewport(self.command_buffer, 0, &viewports);
            d.cmd_set_scissor(self.command_buffer, 0, &scissors);
            d.cmd_bind_vertex_buffers(self.command_buffer, 0, &[self.vertex_buffer], &[0]);
            d.cmd_draw(self.command_buffer, TRIANGLE_VERTICES.len() as u32, 1, 0, 0);
            d.cmd_end_rendering(self.command_buffer);
        }

        // COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL for the copy out.
        self.image_barrier(
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        );

        // buffer_row_length/image_height = 0 means "tightly packed", which is
        // what `RenderedImage::pixel` assumes.
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        // SAFETY: the image is in TRANSFER_SRC_OPTIMAL per the barrier above,
        // and the readback buffer was sized `width * height * 4` — exactly the
        // region copied.
        unsafe {
            self.device().cmd_copy_image_to_buffer(
                self.command_buffer,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                &[region],
            );
        }

        // Make the transfer writes visible to host reads of the mapped memory.
        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: still recording; a global memory barrier names no handles.
        unsafe {
            self.device().cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host_barrier],
                &[],
                &[],
            );
            self.device()
                .end_command_buffer(self.command_buffer)
                .map_err(|e| GpuError::VulkanInitFailed(format!("vkEndCommandBuffer: {e}")))?;
        }

        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);

        // SAFETY: the command buffer is recorded and not already pending; the
        // fence is unsignaled and unused; the queue came from this device.
        unsafe {
            self.device()
                .queue_submit(self.dev.queue(), &[submit], self.fence)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkQueueSubmit failed: {e}")))?;

        // Wait for the GPU. u64::MAX = no timeout; a hang here means a driver
        // fault, which surfaces as a hung test rather than silent bad pixels.
        // SAFETY: the fence was just submitted with and belongs to this device.
        unsafe { self.device().wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkWaitForFences failed: {e}")))?;
        Ok(())
    }

    fn read_back(&self, width: u32, height: u32) -> Result<Vec<u8>, GpuError> {
        let size = (width as usize) * (height as usize) * (BYTES_PER_PIXEL as usize);

        // SAFETY: the memory is HOST_VISIBLE and not currently mapped. The GPU
        // work that writes it has completed — `record_and_submit` waited on the
        // fence — and a barrier made those writes available to the host.
        let ptr = unsafe {
            self.device().map_memory(
                self.readback_memory,
                0,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("readback map failed: {e}")))?;

        // SAFETY: `ptr` is a valid mapping of `size` bytes (the buffer was
        // allocated at exactly this size), initialized by the completed copy.
        // The bytes are read into an owned Vec before unmapping, so no
        // reference outlives the mapping.
        let pixels = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), size).to_vec() };

        // SAFETY: the memory is currently mapped by the call above and no
        // references into it remain.
        unsafe { self.device().unmap_memory(self.readback_memory) };
        Ok(pixels)
    }
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created from `self.dev`'s device and is
        // destroyed exactly once, children before parents. `device_wait_idle`
        // ensures no submitted work still references them; its error is ignored
        // because drop must not panic and a lost device cannot be recovered.
        // Null handles are skipped, so a partially-built `Resources` (an error
        // during `build`) cleans up correctly.
        unsafe {
            let d = self.device();
            let _ = d.device_wait_idle();

            if self.fence != vk::Fence::null() {
                d.destroy_fence(self.fence, None);
            }
            if self.command_buffer != vk::CommandBuffer::null() {
                d.free_command_buffers(self.dev.command_pool(), &[self.command_buffer]);
            }
            if self.pipeline != vk::Pipeline::null() {
                d.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                d.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.fragment_module != vk::ShaderModule::null() {
                d.destroy_shader_module(self.fragment_module, None);
            }
            if self.vertex_module != vk::ShaderModule::null() {
                d.destroy_shader_module(self.vertex_module, None);
            }
            if self.readback_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.readback_buffer, None);
            }
            if self.readback_memory != vk::DeviceMemory::null() {
                d.free_memory(self.readback_memory, None);
            }
            if self.vertex_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.vertex_buffer, None);
            }
            if self.vertex_memory != vk::DeviceMemory::null() {
                d.free_memory(self.vertex_memory, None);
            }
            if self.image_view != vk::ImageView::null() {
                d.destroy_image_view(self.image_view, None);
            }
            if self.image != vk::Image::null() {
                d.destroy_image(self.image, None);
            }
            if self.image_memory != vk::DeviceMemory::null() {
                d.free_memory(self.image_memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::shaders::TRIANGLE_COLOR;
    use super::*;

    #[test]
    fn unorm8_maps_endpoints_exactly() {
        assert_eq!(unorm8([0.0, 1.0, 0.0, 1.0]), [0, 255, 0, 255]);
        assert_eq!(unorm8(TRIANGLE_COLOR), [0, 255, 0, 255]);
    }

    #[test]
    fn unorm8_clamps_out_of_range() {
        assert_eq!(unorm8([-1.0, 2.0, 0.5, 1.0]), [0, 255, 128, 255]);
    }

    /// The test asserts corners are clear-colored, which only holds if the
    /// triangle stays away from them.
    #[test]
    fn triangle_covers_center_but_no_corner() {
        let inside = |px: f32, py: f32| {
            let [a, b, c] = TRIANGLE_VERTICES;
            let sign = |p: [f32; 4], q: [f32; 4]| {
                (px - q[0]) * (p[1] - q[1]) - (p[0] - q[0]) * (py - q[1])
            };
            let d1 = sign(a, b);
            let d2 = sign(b, c);
            let d3 = sign(c, a);
            let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
            let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
            !(neg && pos)
        };
        assert!(inside(0.0, 0.0), "NDC center must be covered");
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            assert!(!inside(x, y), "NDC corner ({x}, {y}) must stay uncovered");
        }
    }

    #[test]
    fn rendered_image_indexes_rows_major() {
        let img = RenderedImage {
            width: 2,
            height: 2,
            pixels: vec![
                1, 1, 1, 1, 2, 2, 2, 2, // row 0
                3, 3, 3, 3, 4, 4, 4, 4, // row 1
            ],
        };
        assert_eq!(img.pixel(0, 0), Some([1, 1, 1, 1]));
        assert_eq!(img.pixel(1, 0), Some([2, 2, 2, 2]));
        assert_eq!(img.pixel(0, 1), Some([3, 3, 3, 3]));
        assert_eq!(img.pixel(1, 1), Some([4, 4, 4, 4]));
        assert_eq!(img.pixel(2, 0), None);
        assert_eq!(img.pixel(0, 2), None);
    }
}
