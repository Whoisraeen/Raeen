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
use tracing::debug;
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

/// One guest vertex-buffer binding uploaded for a draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBufferData {
    pub bytes: Vec<u8>,
    pub stride: u32,
}

/// Vulkan's interpretation of one analyzed guest vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttributeData {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

/// One descriptor binding containing an array of storage buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageBufferBinding {
    pub binding: u32,
    pub buffers: Vec<Vec<u8>>,
}

/// Per-stage resource ABI used by translated SPIR-V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderStageBinding {
    pub stage: vk::ShaderStageFlags,
    pub descriptor_set_slot: u32,
    pub push_constant_offset: u32,
    pub push_constants: Vec<u8>,
    pub storage_buffers: Option<StorageBufferBinding>,
}

/// Everything one offscreen draw needs, with nothing hardcoded.
///
/// This is the parameter object [`render_draw`] takes so a caller can drive a
/// draw from decoded PM4 register state instead of the fixture constants.
/// `render_triangle_with_spirv` is now just a preset of this.
#[derive(Debug, Clone)]
pub struct DrawState<'a> {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub clear_color: [f32; 4],
    /// `[left, top, right, bottom]` in pixels.
    pub scissor: [i32; 4],
    /// `[x, y, width, height]` in pixels.
    pub viewport: [f32; 4],
    pub topology: vk::PrimitiveTopology,
    pub cull_mode: vk::CullModeFlags,
    /// Which colour channels the draw may write, from `CB_TARGET_MASK`.
    ///
    /// Vulkan expresses this natively, so a guest mask maps straight through
    /// rather than being a limitation: a title that writes RGB and leaves alpha
    /// alone (mask 0x7) is doing something completely ordinary.
    pub color_write_mask: vk::ColorComponentFlags,
    /// Host vertex data, or `None` for a shader that synthesizes its own
    /// geometry from `gl_VertexIndex` (Kyty's embedded VS does exactly this and
    /// declares no input attributes, so binding one would be invalid).
    pub vertices: Option<&'a [[f32; 4]]>,
    /// Guest-backed vertex buffers and translated Vulkan attributes.
    pub vertex_buffers: Vec<VertexBufferData>,
    pub vertex_attributes: Vec<VertexAttributeData>,
    /// Descriptor sets and push constants required by translated stages.
    pub stage_bindings: Vec<ShaderStageBinding>,
    pub vertex_count: u32,
    pub vs_spirv: &'a [u32],
    pub fs_spirv: &'a [u32],
    /// Prior contents of the render target (tightly-packed RGBA,
    /// `width * height * 4` bytes). When present the attachment is seeded
    /// with these pixels and `load_op` is LOAD instead of CLEAR — this is
    /// how successive draws into the same guest render target compose into
    /// one frame across otherwise independent one-shot draw submissions.
    pub initial: Option<&'a [u8]>,
}

impl<'a> DrawState<'a> {
    /// A full-target viewport/scissor at `width` x `height`, no vertex buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, vs_spirv: &'a [u32], fs_spirv: &'a [u32]) -> Self {
        Self {
            width,
            height,
            format: vk::Format::R8G8B8A8_UNORM,
            clear_color: CLEAR_COLOR,
            scissor: [0, 0, width as i32, height as i32],
            viewport: [0.0, 0.0, width as f32, height as f32],
            // The fixture preset writes every channel; a guest draw overrides
            // this from CB_TARGET_MASK.
            color_write_mask: vk::ColorComponentFlags::RGBA,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            cull_mode: vk::CullModeFlags::NONE,
            vertices: None,
            vertex_buffers: Vec::new(),
            vertex_attributes: Vec::new(),
            stage_bindings: Vec::new(),
            vertex_count: 0,
            vs_spirv,
            fs_spirv,
            initial: None,
        }
    }
}

/// `RenderedImage` indexes pixels assuming 4 bytes each, so a format of any
/// other size would silently corrupt the readback rather than fail.
fn require_32bpp(format: vk::Format) -> Result<(), GpuError> {
    match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB => Ok(()),
        other => Err(GpuError::VulkanInitFailed(format!(
            "render target format {other:?} is not 32bpp; readback assumes {BYTES_PER_PIXEL} bytes per pixel"
        ))),
    }
}

/// Draw once offscreen from an explicit [`DrawState`] and read back the pixels.
///
/// # Errors
///
/// [`GpuError::VulkanInitFailed`] on a zero-sized or non-32bpp target or any
/// resource/submission failure, [`GpuError::ShaderCompilationFailed`] on empty
/// SPIR-V, [`GpuError::PipelineCreationFailed`] if the pipeline is rejected.
pub fn render_draw(dev: &VulkanDevice, state: &DrawState) -> Result<RenderedImage, GpuError> {
    if state.width == 0 || state.height == 0 {
        return Err(GpuError::VulkanInitFailed(format!(
            "invalid render target size {}x{}",
            state.width, state.height
        )));
    }
    if state.vs_spirv.is_empty() || state.fs_spirv.is_empty() {
        return Err(GpuError::ShaderCompilationFailed(
            "vertex and fragment SPIR-V must be non-empty".to_owned(),
        ));
    }
    require_32bpp(state.format)?;

    let mut res = Resources::new(dev);
    res.build(state)?;
    res.record_and_submit(state)?;
    let pixels = res.read_back(state.width, state.height)?;

    debug!(
        width = state.width,
        height = state.height,
        vertices = state.vertex_count,
        "offscreen draw rendered on {}",
        dev.device_name()
    );
    Ok(RenderedImage {
        width: state.width,
        height: state.height,
        pixels,
    })
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
    render_triangle_with_spirv(
        dev,
        width,
        height,
        &triangle_vertex_spirv(),
        &triangle_fragment_spirv(),
    )
}

/// Draw one triangle offscreen using caller-supplied SPIR-V modules.
///
/// Used by the M2 path (`kyty-graphics` SPIR-V) and by the hand-built smoke
/// path via [`render_triangle`].
pub fn render_triangle_with_spirv(
    dev: &VulkanDevice,
    width: u32,
    height: u32,
    vs_spirv: &[u32],
    fs_spirv: &[u32],
) -> Result<RenderedImage, GpuError> {
    render_draw(
        dev,
        &DrawState {
            vertices: Some(&TRIANGLE_VERTICES),
            vertex_count: TRIANGLE_VERTICES.len() as u32,
            ..DrawState::new(width, height, vs_spirv, fs_spirv)
        },
    )
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
    guest_vertex_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    storage_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<(u32, vk::DescriptorSet)>,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
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
            guest_vertex_buffers: Vec::new(),
            storage_buffers: Vec::new(),
            descriptor_set_layouts: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            upload_buffer: vk::Buffer::null(),
            upload_memory: vk::DeviceMemory::null(),
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

    fn build(&mut self, state: &DrawState) -> Result<(), GpuError> {
        self.create_render_target(state.width, state.height, state.format)?;
        if let Some(initial) = state.initial {
            let expected = state.width as usize * state.height as usize * BYTES_PER_PIXEL as usize;
            if initial.len() != expected {
                return Err(GpuError::VulkanInitFailed(format!(
                    "initial render-target contents are {} bytes; {}x{} needs {expected}",
                    initial.len(),
                    state.width,
                    state.height
                )));
            }
            let (buffer, memory) =
                self.create_buffer_with_bytes(initial, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.upload_buffer = buffer;
            self.upload_memory = memory;
        }
        if let Some(vertices) = state.vertices {
            self.create_vertex_buffer(vertices)?;
        }
        self.create_guest_vertex_buffers(state)?;
        self.create_stage_resources(state)?;
        self.create_readback_buffer(state.width, state.height)?;
        self.create_pipeline(state)?;
        self.create_command_resources()?;
        Ok(())
    }

    fn create_render_target(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(), GpuError> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // COLOR_ATTACHMENT to draw into it, TRANSFER_SRC to copy it out,
            // TRANSFER_DST to seed it with the target's prior contents.
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
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
            .format(format)
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

    fn create_vertex_buffer(&mut self, vertices: &[[f32; 4]]) -> Result<(), GpuError> {
        let size = mem::size_of_val(vertices) as vk::DeviceSize;
        if size == 0 {
            return Err(GpuError::VulkanInitFailed(
                "vertex buffer requested with no vertices".to_owned(),
            ));
        }
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
            align.copy_from_slice(vertices);
            self.device().unmap_memory(memory);
        }
        Ok(())
    }

    fn create_buffer_with_bytes(
        &self,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        if bytes.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "buffer upload requested with no bytes".to_owned(),
            ));
        }
        let size = bytes.len() as vk::DeviceSize;
        let (buffer, memory) = self.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory failed: {e}")))?;

        // SAFETY: `memory` is HOST_VISIBLE and mapped for exactly
        // `bytes.len()` bytes. No GPU submission can reference it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.device().unmap_memory(memory);
        }
        Ok((buffer, memory))
    }

    fn create_guest_vertex_buffers(&mut self, state: &DrawState) -> Result<(), GpuError> {
        for vertex in &state.vertex_buffers {
            if vertex.stride == 0 || vertex.bytes.len() < vertex.stride as usize {
                return Err(GpuError::VulkanInitFailed(format!(
                    "invalid guest vertex buffer: {} bytes, stride {}",
                    vertex.bytes.len(),
                    vertex.stride
                )));
            }
            let allocation =
                self.create_buffer_with_bytes(&vertex.bytes, vk::BufferUsageFlags::VERTEX_BUFFER)?;
            self.guest_vertex_buffers.push(allocation);
        }
        Ok(())
    }

    fn create_stage_resources(&mut self, state: &DrawState) -> Result<(), GpuError> {
        let descriptor_stages: Vec<_> = state
            .stage_bindings
            .iter()
            .filter(|stage| stage.storage_buffers.is_some())
            .collect();
        if descriptor_stages.is_empty() {
            return Ok(());
        }

        for stage in &descriptor_stages {
            if stage.descriptor_set_slot as usize != self.descriptor_set_layouts.len() {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "descriptor set slot {} is not contiguous (expected {})",
                    stage.descriptor_set_slot,
                    self.descriptor_set_layouts.len()
                )));
            }
            let storage = stage.storage_buffers.as_ref().expect("filtered above");
            if storage.buffers.is_empty() {
                return Err(GpuError::PipelineCreationFailed(
                    "storage-buffer descriptor array is empty".to_owned(),
                ));
            }
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(storage.binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(storage.buffers.len() as u32)
                .stage_flags(stage.stage)];
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            // SAFETY: `bindings` remains alive for the call; the returned
            // layout is retained through pipeline and descriptor-set use.
            let layout = unsafe { self.device().create_descriptor_set_layout(&info, None) }
                .map_err(|e| {
                    GpuError::PipelineCreationFailed(format!("vkCreateDescriptorSetLayout: {e}"))
                })?;
            self.descriptor_set_layouts.push(layout);
        }

        let total_storage: u32 = descriptor_stages
            .iter()
            .map(|stage| {
                stage
                    .storage_buffers
                    .as_ref()
                    .map_or(0, |storage| storage.buffers.len() as u32)
            })
            .sum();
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(total_storage)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(descriptor_stages.len() as u32)
            .pool_sizes(&pool_sizes);
        // SAFETY: the pool-size slice is alive for the call. Pool lifetime is
        // owned by this resource bundle and outlives all allocated sets.
        self.descriptor_pool = unsafe { self.device().create_descriptor_pool(&pool_info, None) }
            .map_err(|e| {
                GpuError::PipelineCreationFailed(format!("vkCreateDescriptorPool: {e}"))
            })?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&self.descriptor_set_layouts);
        // SAFETY: the pool and every layout are live handles from this device.
        let sets = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }.map_err(|e| {
            GpuError::PipelineCreationFailed(format!("vkAllocateDescriptorSets: {e}"))
        })?;

        for (stage, set) in descriptor_stages.into_iter().zip(sets) {
            let storage = stage.storage_buffers.as_ref().expect("filtered above");
            let first_buffer = self.storage_buffers.len();
            for bytes in &storage.buffers {
                let allocation =
                    self.create_buffer_with_bytes(bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
                self.storage_buffers.push(allocation);
            }
            let infos: Vec<_> = self.storage_buffers[first_buffer..]
                .iter()
                .map(|(buffer, _)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(*buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(storage.binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos)];
            // SAFETY: `set` came from our pool/layout and `infos` names live
            // buffers retained by this resource bundle.
            unsafe { self.device().update_descriptor_sets(&writes, &[]) };
            self.descriptor_sets.push((stage.descriptor_set_slot, set));
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

    fn create_pipeline(&mut self, state: &DrawState) -> Result<(), GpuError> {
        self.vertex_module = self.create_shader_module(state.vs_spirv)?;
        self.fragment_module = self.create_shader_module(state.fs_spirv)?;

        let push_ranges: Vec<_> = state
            .stage_bindings
            .iter()
            .filter(|stage| !stage.push_constants.is_empty())
            .map(|stage| {
                vk::PushConstantRange::default()
                    .stage_flags(stage.stage)
                    .offset(stage.push_constant_offset)
                    .size(stage.push_constants.len() as u32)
            })
            .collect();
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&self.descriptor_set_layouts)
            .push_constant_ranges(&push_ranges);
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

        // One vec4 attribute at location 0, matching the vertex shader — but
        // only when the caller supplies vertices. A shader that builds its
        // geometry from `gl_VertexIndex` declares no inputs, and binding an
        // attribute it never consumes is invalid.
        let fixture_bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<[f32; 4]>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let fixture_attributes = [vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(0)];
        let guest_bindings: Vec<_> = state
            .vertex_buffers
            .iter()
            .enumerate()
            .map(|(binding, data)| {
                vk::VertexInputBindingDescription::default()
                    .binding(binding as u32)
                    .stride(data.stride)
                    .input_rate(vk::VertexInputRate::VERTEX)
            })
            .collect();
        let guest_attributes: Vec<_> = state
            .vertex_attributes
            .iter()
            .map(|attr| {
                vk::VertexInputAttributeDescription::default()
                    .location(attr.location)
                    .binding(attr.binding)
                    .format(attr.format)
                    .offset(attr.offset)
            })
            .collect();
        let vertex_input = if !state.vertex_buffers.is_empty() {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&guest_bindings)
                .vertex_attribute_descriptions(&guest_attributes)
        } else if state.vertices.is_some() {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&fixture_bindings)
                .vertex_attribute_descriptions(&fixture_attributes)
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
        };

        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(state.topology);

        // Viewport and scissor are dynamic, set during recording.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(state.cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(state.color_write_mask)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Vulkan 1.3 dynamic rendering: the pipeline declares the attachment
        // formats directly instead of referencing a VkRenderPass.
        let color_formats = [state.format];
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

    fn record_and_submit(&mut self, state: &DrawState) -> Result<(), GpuError> {
        let (width, height) = (state.width, state.height);
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: the command buffer is freshly allocated and not pending, so
        // beginning it is legal; it is recorded only from this thread.
        unsafe {
            self.device()
                .begin_command_buffer(self.command_buffer, &begin_info)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkBeginCommandBuffer: {e}")))?;

        if state.initial.is_some() {
            // Seed the attachment with the target's prior contents:
            // UNDEFINED -> TRANSFER_DST, copy in, then hand off to the
            // attachment stage so LOAD sees the composed frame so far.
            self.image_barrier(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            );
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
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            // SAFETY: the upload buffer holds exactly width*height*4 bytes
            // (validated in build) and the image was created TRANSFER_DST;
            // both belong to this device and the command buffer is recording.
            unsafe {
                self.device().cmd_copy_buffer_to_image(
                    self.command_buffer,
                    self.upload_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
            self.image_barrier(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            );
        } else {
            // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL. Discards existing
            // contents, which is fine: the render pass clears anyway.
            self.image_barrier(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            );
        }

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: state.clear_color,
            },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(if state.initial.is_some() {
                vk::AttachmentLoadOp::LOAD
            } else {
                vk::AttachmentLoadOp::CLEAR
            })
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
            x: state.viewport[0],
            y: state.viewport[1],
            width: state.viewport[2],
            height: state.viewport[3],
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        // Clamp the register-supplied scissor into the attachment: Vulkan
        // rejects a scissor that leaves the render area, and a guest is free to
        // program one that does.
        let [sl, st, sr, sb] = state.scissor;
        let sl = sl.clamp(0, width as i32);
        let st = st.clamp(0, height as i32);
        let sr = sr.clamp(sl, width as i32);
        let sb = sb.clamp(st, height as i32);
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: sl, y: st },
            extent: vk::Extent2D {
                width: (sr - sl) as u32,
                height: (sb - st) as u32,
            },
        }];

        // SAFETY: all handles below belong to this device and are live; the
        // command buffer is recording; the vertex buffer (when bound) holds
        // exactly `state.vertex_count` vertices, matching the draw; the
        // pipeline's attachment format matches the image view's.
        // `cmd_begin_rendering` is core in Vulkan 1.3 and `dynamicRendering`
        // was required at device selection.
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
            if !self.guest_vertex_buffers.is_empty() {
                let buffers: Vec<_> = self
                    .guest_vertex_buffers
                    .iter()
                    .map(|(buffer, _)| *buffer)
                    .collect();
                let offsets = vec![0; buffers.len()];
                d.cmd_bind_vertex_buffers(self.command_buffer, 0, &buffers, &offsets);
            } else if state.vertices.is_some() {
                d.cmd_bind_vertex_buffers(self.command_buffer, 0, &[self.vertex_buffer], &[0]);
            }
            for stage in &state.stage_bindings {
                if let Some((_, set)) = self
                    .descriptor_sets
                    .iter()
                    .find(|(slot, _)| *slot == stage.descriptor_set_slot)
                {
                    d.cmd_bind_descriptor_sets(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        stage.descriptor_set_slot,
                        &[*set],
                        &[],
                    );
                }
                if !stage.push_constants.is_empty() {
                    d.cmd_push_constants(
                        self.command_buffer,
                        self.pipeline_layout,
                        stage.stage,
                        stage.push_constant_offset,
                        &stage.push_constants,
                    );
                }
            }
            d.cmd_draw(self.command_buffer, state.vertex_count, 1, 0, 0);
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
        let descriptor_set_layouts = mem::take(&mut self.descriptor_set_layouts);
        let guest_vertex_buffers = mem::take(&mut self.guest_vertex_buffers);
        let storage_buffers = mem::take(&mut self.storage_buffers);
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
            if self.descriptor_pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            for layout in descriptor_set_layouts {
                d.destroy_descriptor_set_layout(layout, None);
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
            if self.upload_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.upload_buffer, None);
            }
            if self.upload_memory != vk::DeviceMemory::null() {
                d.free_memory(self.upload_memory, None);
            }
            if self.vertex_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.vertex_buffer, None);
            }
            if self.vertex_memory != vk::DeviceMemory::null() {
                d.free_memory(self.vertex_memory, None);
            }
            for (buffer, memory) in guest_vertex_buffers {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            for (buffer, memory) in storage_buffers {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
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
