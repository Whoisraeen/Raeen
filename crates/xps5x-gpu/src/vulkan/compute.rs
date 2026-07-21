//! One-shot Vulkan compute dispatch for translated guest shaders.
//!
//! Guest storage buffers are uploaded into host-visible coherent Vulkan
//! buffers, and guest storage images (UAVs) into device-local
//! `R8G8B8A8_UNORM` images via staging buffers. After the queue fence both
//! are read back; the caller owns copying those bytes back into
//! identity-mapped guest memory.

use super::cache::DrawCaches;
use super::instance::VulkanDevice;
use super::offscreen::{ShaderStageBinding, StorageImageUpload, TextureUpload};
use ash::vk::Handle;
use ash::{Device, vk};
use xps5x_core::error::GpuError;

pub struct ComputeState<'a> {
    pub groups: [u32; 3],
    pub spirv: &'a [u32],
    pub binding: Option<&'a ShaderStageBinding>,
}

/// Post-dispatch device content, in the binding's declaration order.
pub struct ComputeOutputs {
    /// One entry per storage buffer (`StorageBufferBinding.buffers` order).
    pub buffers: Vec<Vec<u8>>,
    /// One entry per storage image (`StorageImageBinding.images` order),
    /// RGBA8 tightly packed rows.
    pub images: Vec<Vec<u8>>,
}

pub fn dispatch_compute(
    dev: &VulkanDevice,
    state: &ComputeState<'_>,
) -> Result<ComputeOutputs, GpuError> {
    if state.spirv.is_empty() {
        return Err(GpuError::ShaderCompilationFailed(
            "compute SPIR-V must be non-empty".to_owned(),
        ));
    }
    if let Some(binding) = state.binding
        && binding.stage != vk::ShaderStageFlags::COMPUTE
    {
        return Err(GpuError::PipelineCreationFailed(format!(
            "compute binding has non-compute stage {:?}",
            binding.stage
        )));
    }

    // Same locking contract as `render_draw`: the cache lock spans the whole
    // synchronous dispatch, including the fence wait, so the cached pipeline,
    // command buffer, fence, and descriptor pool are reused soundly.
    let mut caches = dev.draw_caches();
    let mut resources = ComputeResources::new(dev, &mut caches);
    resources.build(state)?;
    resources.record_and_submit(state)?;
    Ok(ComputeOutputs {
        buffers: resources.read_storage()?,
        images: resources.read_images()?,
    })
}

struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
}

/// One storage image plus its upload staging and readback buffers.
struct ImageAllocation {
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    /// Volume depth (1 for a 2D UAV).
    depth: u32,
    /// Bytes per texel (4 = RGBA8, 8 = RGBA16F).
    texel: u32,
}

/// One sampled (read-only) texture: staging buffer + device-local image +
/// view. Uploaded once before the dispatch; never read back.
struct SampledAllocation {
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
}

/// Per-dispatch resources (owned, destroyed in `Drop`) plus handles borrowed
/// from [`DrawCaches`] — shader module, descriptor layout, descriptor pool,
/// pipeline layout, pipeline, samplers, command buffer, and fence are cached
/// on the device and must NOT be destroyed here.
struct ComputeResources<'a> {
    dev: &'a VulkanDevice,
    caches: &'a mut DrawCaches,
    storage: Vec<BufferAllocation>,
    images: Vec<ImageAllocation>,
    sampled: Vec<SampledAllocation>,
    samplers: Vec<vk::Sampler>,
    /// The persistent GDS arena (cache-owned, NOT destroyed here); null when
    /// the dispatch binds no GDS.
    gds: vk::Buffer,
    /// The raw EUD-window snapshot (SharpEmu port); per-dispatch, owned,
    /// never read back.
    eud_raw: Option<BufferAllocation>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl<'a> ComputeResources<'a> {
    fn new(dev: &'a VulkanDevice, caches: &'a mut DrawCaches) -> Self {
        Self {
            dev,
            caches,
            storage: Vec::new(),
            images: Vec::new(),
            sampled: Vec::new(),
            samplers: Vec::new(),
            gds: vk::Buffer::null(),
            eud_raw: None,
            shader: vk::ShaderModule::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        }
    }

    fn device(&self) -> &Device {
        self.dev.device()
    }

    fn build(&mut self, state: &ComputeState<'_>) -> Result<(), GpuError> {
        // Cached by SPIR-V content: a title re-dispatches the same compute
        // shaders thousands of times per frame.
        self.shader = self.caches.shader_module(self.dev, state.spirv)?;

        let storage = state.binding.and_then(|b| b.storage_buffers.as_ref());
        let storage_images = state.binding.and_then(|b| b.storage_images.as_ref());
        let textures = state.binding.and_then(|b| b.textures.as_ref());
        let gds_binding = state.binding.and_then(|b| b.gds_binding);
        let eud_raw = state.binding.and_then(|b| b.eud_raw.as_ref());
        if storage.is_some()
            || storage_images.is_some()
            || textures.is_some()
            || gds_binding.is_some()
            || eud_raw.is_some()
        {
            let binding = state.binding.expect("resource groups come from a binding");
            if binding.descriptor_set_slot != 0 {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "compute descriptor set slot {} is not supported yet",
                    binding.descriptor_set_slot
                )));
            }
            let mut layout_bindings = Vec::new();
            if let Some(storage) = storage {
                if storage.buffers.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "compute storage descriptor array is empty".to_owned(),
                    ));
                }
                for bytes in &storage.buffers {
                    self.storage.push(self.create_storage_buffer(bytes)?);
                }
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(storage.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(self.storage.len() as u32)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            if let Some(images) = storage_images {
                if images.images.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "compute storage-image descriptor array is empty".to_owned(),
                    ));
                }
                for upload in &images.images {
                    self.create_storage_image(upload)?;
                }
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(images.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(self.images.len() as u32)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            // Sampled textures + samplers (the recompiled SPIR-V declares
            // %textures2D_S and %samplers as separate bindings) — first
            // consumed by ASTRO.BOT's froxel/LUT-volume compute shaders.
            //
            // The two arrays are INDEPENDENT: the SPIR-V declares
            // `%textures2D_S` only when `textures2d_sampled_num > 0` and
            // `%samplers` only when `samplers_num > 0`, and a CS legitimately
            // carries one without the other (texel-fetch/image-load shaders
            // bind textures but zero samplers). Each descriptor array is
            // created only when non-empty, mirroring the SPIR-V exactly.
            if let Some(textures) = textures {
                if textures.textures.is_empty() && textures.linear_filter.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "compute sampled-texture and sampler descriptor arrays are both empty"
                            .to_owned(),
                    ));
                }
                if !textures.textures.is_empty() {
                    for upload in &textures.textures {
                        self.create_sampled_image(upload)?;
                    }
                    if textures.sampled_groups.is_empty() {
                        // Homogeneous: one array of every sampled view.
                        layout_bindings.push(
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(textures.sampled_binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(self.sampled.len() as u32)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        );
                    } else {
                        // Mixed-dim: one `%textures2D_S<dim>` array per Dim, at
                        // its own binding — matching the recompiled SPIR-V.
                        for group in &textures.sampled_groups {
                            layout_bindings.push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(group.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .descriptor_count(group.view_indices.len() as u32)
                                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                            );
                        }
                    }
                }
                if !textures.linear_filter.is_empty() {
                    for &linear in &textures.linear_filter {
                        self.samplers.push(self.caches.sampler(self.dev, linear)?);
                    }
                    layout_bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(self.samplers.len() as u32)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    );
                }
            }
            // The persistent GDS arena: one storage buffer whose contents
            // persist across dispatches (device lifetime). Bound at the
            // binding index `shader_calc_binding_indices` assigned to
            // `%gds`; never read back (GDS is on-chip, not guest memory).
            if let Some(binding) = gds_binding {
                self.gds = self.caches.gds_buffer(self.dev)?;
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            // The raw EUD-window snapshot (SharpEmu port): one per-dispatch
            // SSBO at the binding index the recompiled `%eud_raw` declares.
            // Uploaded once, never read back (the shader only s_loads it).
            if let Some(window) = eud_raw {
                if window.bytes.len() < 4 || !window.bytes.len().is_multiple_of(4) {
                    return Err(GpuError::PipelineCreationFailed(format!(
                        "raw EUD-window snapshot carries {} bytes — must be a non-zero \
                         dword multiple",
                        window.bytes.len()
                    )));
                }
                self.eud_raw = Some(self.create_storage_buffer(&window.bytes)?);
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(window.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }

            // Cached by binding signature, shared with the graphics path.
            self.descriptor_layout = self.caches.set_layout(self.dev, &layout_bindings)?;

            let pool_sizes: Vec<_> = [
                (
                    vk::DescriptorType::STORAGE_BUFFER,
                    self.storage.len() as u32
                        + u32::from(!self.gds.is_null())
                        + u32::from(self.eud_raw.is_some()),
                ),
                (vk::DescriptorType::STORAGE_IMAGE, self.images.len() as u32),
                (vk::DescriptorType::SAMPLED_IMAGE, self.sampled.len() as u32),
                (vk::DescriptorType::SAMPLER, self.samplers.len() as u32),
            ]
            .into_iter()
            .filter(|&(_, count)| count != 0)
            .map(|(ty, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(ty)
                    .descriptor_count(count)
            })
            .collect();
            // The persistent pool, reset for this dispatch (the previous
            // draw/dispatch that used it fence-completed under the lock).
            self.descriptor_pool = self.caches.descriptor_pool(self.dev, 1, &pool_sizes)?;
            let layouts = [self.descriptor_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);
            // SAFETY: pool and layout are live handles from this device.
            self.descriptor_set = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }
                .map_err(|e| {
                    GpuError::PipelineCreationFailed(format!("vkAllocateDescriptorSets: {e}"))
                })?[0];

            let buffer_infos: Vec<_> = self
                .storage
                .iter()
                .map(|allocation| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(allocation.buffer)
                        .range(allocation.size as u64)
                })
                .collect();
            let image_infos: Vec<_> = self
                .images
                .iter()
                .map(|allocation| {
                    vk::DescriptorImageInfo::default()
                        .image_view(allocation.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();
            let mut writes = Vec::new();
            if let Some(storage) = storage {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(storage.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_infos),
                );
            }
            if let Some(images) = storage_images {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(images.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image_infos),
                );
            }
            let sampled_infos: Vec<_> = self
                .sampled
                .iter()
                .map(|allocation| {
                    vk::DescriptorImageInfo::default()
                        .image_view(allocation.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();
            let sampler_infos: Vec<_> = self
                .samplers
                .iter()
                .map(|&sampler| vk::DescriptorImageInfo::default().sampler(sampler))
                .collect();
            // Mixed-dim: split the sampled-view pool into one descriptor array
            // per Dim, in SPIR-V array order. `self.sampled[i]` corresponds to
            // `textures.textures[i]`, so a group's `view_indices` select its
            // views directly. Kept alive alongside `sampled_infos`.
            let group_infos: Vec<Vec<vk::DescriptorImageInfo>> = textures
                .map(|t| {
                    t.sampled_groups
                        .iter()
                        .map(|group| {
                            group
                                .view_indices
                                .iter()
                                .map(|&i| sampled_infos[i])
                                .collect()
                        })
                        .collect()
                })
                .unwrap_or_default();
            if let Some(textures) = textures {
                if !sampled_infos.is_empty() {
                    if textures.sampled_groups.is_empty() {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(self.descriptor_set)
                                .dst_binding(textures.sampled_binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&sampled_infos),
                        );
                    } else {
                        for (group, infos) in textures.sampled_groups.iter().zip(&group_infos) {
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_set)
                                    .dst_binding(group.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(infos),
                            );
                        }
                    }
                }
                if !sampler_infos.is_empty() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler_infos),
                    );
                }
            }
            let gds_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.gds)
                .range(super::cache::GDS_SIZE as u64)];
            if let Some(binding) = gds_binding {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&gds_info),
                );
            }
            let eud_raw_info = self
                .eud_raw
                .as_ref()
                .map(|allocation| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(allocation.buffer)
                        .range(allocation.size as u64)]
                })
                .unwrap_or_default();
            if let Some(window) = eud_raw {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(window.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&eud_raw_info),
                );
            }
            // SAFETY: descriptor set and every named buffer/view are live.
            unsafe { self.device().update_descriptor_sets(&writes, &[]) };
        }

        let set_layouts: Vec<_> = (!self.descriptor_layout.is_null())
            .then_some(self.descriptor_layout)
            .into_iter()
            .collect();
        // Exceeding maxPushConstantsSize is invalid usage — UB without
        // validation layers (measured: AMD driver access violation). Refuse
        // the dispatch by name until the SSBO spill path exists; iGPUs
        // commonly report the 256-byte spec minimum.
        if let Some(binding) = state.binding {
            let need = binding.push_constant_offset + binding.push_constants.len() as u32;
            let cap = self.dev.max_push_constants_size();
            if need > cap {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "push constants {need} B exceed the device maxPushConstantsSize {cap} B \
                     (SSBO spill not implemented)"
                )));
            }
        }
        let push_ranges: Vec<_> = state
            .binding
            .filter(|binding| !binding.push_constants.is_empty())
            .map(|binding| {
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(binding.push_constant_offset)
                    .size(binding.push_constants.len() as u32)
            })
            .into_iter()
            .collect();
        self.pipeline_layout = self
            .caches
            .pipeline_layout(self.dev, &set_layouts, &push_ranges)?;

        // Cached by (canonical module, canonical layout): identical dispatch
        // programs reuse one VkPipeline instead of recompiling per dispatch.
        self.pipeline =
            self.caches
                .compute_pipeline(self.dev, self.shader, self.pipeline_layout)?;

        let (command_buffer, fence) = self.caches.submit_resources(self.dev)?;
        self.command_buffer = command_buffer;
        self.fence = fence;
        Ok(())
    }

    /// Host-visible coherent buffer, optionally filled with `fill`.
    fn create_host_buffer(
        &self,
        size: usize,
        usage: vk::BufferUsageFlags,
        fill: Option<&[u8]>,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        if size == 0 {
            return Err(GpuError::VulkanInitFailed(
                "zero-sized compute host buffer".to_owned(),
            ));
        }
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: create info is local and the device is live.
        let buffer = unsafe { self.device().create_buffer(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateBuffer: {e}")))?;
        // SAFETY: buffer is a live handle from this device.
        let req = unsafe { self.device().get_buffer_memory_requirements(buffer) };
        let memory_type = match self.dev.find_memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(t) => t,
            Err(e) => {
                // SAFETY: destroying the just-created, never-bound buffer.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type);
        // SAFETY: requirements and memory type belong to this buffer/device.
        let memory = match unsafe { self.device().allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: destroying the just-created, never-bound buffer.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(GpuError::VulkanInitFailed(format!("vkAllocateMemory: {e}")));
            }
        };
        // SAFETY: buffer and allocation are compatible live handles.
        if let Err(e) = unsafe { self.device().bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: destroying the just-created buffer and its allocation.
            unsafe {
                self.device().destroy_buffer(buffer, None);
                self.device().free_memory(memory, None);
            }
            return Err(GpuError::VulkanInitFailed(format!(
                "vkBindBufferMemory: {e}"
            )));
        }
        if let Some(bytes) = fill {
            debug_assert_eq!(bytes.len(), size);
            // SAFETY: host-visible allocation is mapped for the filled range.
            let ptr = unsafe {
                self.device()
                    .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())
            }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory: {e}")))?;
            // SAFETY: mapped range is at least `size`; pointers do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
                self.device().unmap_memory(memory);
            }
        }
        Ok((buffer, memory))
    }

    fn create_storage_buffer(&self, bytes: &[u8]) -> Result<BufferAllocation, GpuError> {
        if bytes.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "zero-sized compute storage buffer".to_owned(),
            ));
        }
        let (buffer, memory) = self.create_host_buffer(
            bytes.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Some(bytes),
        )?;
        Ok(BufferAllocation {
            buffer,
            memory,
            size: bytes.len(),
        })
    }

    /// One UAV: staging buffer + device-local image + view + readback
    /// buffer, in the upload's own format; `depth > 1` builds a
    /// `VK_IMAGE_TYPE_3D` volume (measured: ASTRO.BOT's 240x135x64 RGBA16F
    /// UAVs). Pushed with null handles up front so `Drop` cleans up any
    /// partially-built entry on the error paths.
    fn create_storage_image(&mut self, upload: &StorageImageUpload) -> Result<(), GpuError> {
        let depth = upload.depth.max(1);
        let texel = upload.texel_bytes();
        let size = (upload.width as usize)
            * (upload.height as usize)
            * (depth as usize)
            * (texel as usize);
        if size == 0 || upload.pixels.len() != size {
            return Err(GpuError::VulkanInitFailed(format!(
                "storage image {}x{}x{depth} ({texel} B/texel) carries {} initial bytes \
                 (want {size})",
                upload.width,
                upload.height,
                upload.pixels.len()
            )));
        }
        self.images.push(ImageAllocation {
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            width: upload.width,
            height: upload.height,
            depth,
            texel,
        });
        let slot = self.images.len() - 1;

        let (staging_buffer, staging_memory) = self.create_host_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some(&upload.pixels),
        )?;
        self.images[slot].staging_buffer = staging_buffer;
        self.images[slot].staging_memory = staging_memory;

        let info = vk::ImageCreateInfo::default()
            .image_type(if depth > 1 {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            })
            .format(upload.format)
            .extent(vk::Extent3D {
                width: upload.width,
                height: upload.height,
                depth,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live.
        let image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage vkCreateImage: {e}")))?;
        self.images[slot].image = image;

        // SAFETY: `image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(image) };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: allocation size/type come from this image's own requirements.
        let memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage image allocation: {e}")))?;
        self.images[slot].memory = memory;

        // SAFETY: memory was allocated for exactly this image; offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage image bind memory: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(if depth > 1 {
                vk::ImageViewType::TYPE_3D
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(upload.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        let view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage image view: {e}")))?;
        self.images[slot].view = view;

        let (readback_buffer, readback_memory) =
            self.create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST, None)?;
        self.images[slot].readback_buffer = readback_buffer;
        self.images[slot].readback_memory = readback_memory;
        Ok(())
    }

    /// One sampled texture: staging buffer + device-local image + view, in
    /// the upload's own decoded format. `depth > 1` builds a
    /// `VK_IMAGE_TYPE_3D` volume with a `3D` view (measured: ASTRO.BOT's
    /// 240x135x64 froxel/LUT volumes); `cube` a CUBE view over 6 layers.
    fn create_sampled_image(&mut self, upload: &TextureUpload) -> Result<(), GpuError> {
        if upload.pixels.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "compute sampled texture with no pixels".to_owned(),
            ));
        }
        self.sampled.push(SampledAllocation {
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: upload.width,
            height: upload.height,
            depth: upload.depth.max(1),
            layers: upload.layers,
        });
        let slot = self.sampled.len() - 1;

        let (staging_buffer, staging_memory) = self.create_host_buffer(
            upload.pixels.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some(&upload.pixels),
        )?;
        self.sampled[slot].staging_buffer = staging_buffer;
        self.sampled[slot].staging_memory = staging_memory;

        let volume = upload.depth > 1;
        let info = vk::ImageCreateInfo::default()
            .image_type(if volume {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            })
            .format(upload.format)
            .extent(vk::Extent3D {
                width: upload.width,
                height: upload.height,
                depth: upload.depth.max(1),
            })
            .mip_levels(1)
            .array_layers(upload.layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(if upload.cube {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            });
        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live.
        let image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled vkCreateImage: {e}")))?;
        self.sampled[slot].image = image;

        // SAFETY: `image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(image) };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: allocation size/type come from this image's own requirements.
        let memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled image allocation: {e}")))?;
        self.sampled[slot].memory = memory;

        // SAFETY: memory was allocated for exactly this image; offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled image bind memory: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(if upload.cube {
                vk::ImageViewType::CUBE
            } else if volume {
                vk::ImageViewType::TYPE_3D
            } else if upload.layers > 1 {
                // 2DArray (T# type 13) — the recompiled SPIR-V samples an
                // arrayed 2D image (measured: ASTRO.BOT's 1536x1536x3).
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(upload.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: upload.layers,
            });
        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        let view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled image view: {e}")))?;
        self.sampled[slot].view = view;
        Ok(())
    }

    fn record_and_submit(&self, state: &ComputeState<'_>) -> Result<(), GpuError> {
        let full_color = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let copy_region = |allocation: &ImageAllocation| {
            vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: allocation.width,
                    height: allocation.height,
                    depth: allocation.depth,
                })
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the cached command buffer is not pending — its previous
        // submission fence-completed under the cache lock — and the pool was
        // created RESET_COMMAND_BUFFER, so begin implicitly resets it.
        unsafe {
            self.device()
                .begin_command_buffer(self.command_buffer, &begin)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkBeginCommandBuffer: {e}")))?;
        // SAFETY: pipeline/layout/sets/images are live and the command buffer
        // is recording; barriers and copies name handles this bundle retains
        // until after the fence wait.
        unsafe {
            // Upload every UAV's initial content and move it to GENERAL, the
            // layout the STORAGE_IMAGE descriptor promised.
            for allocation in &self.images {
                let to_transfer = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(full_color);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                self.device().cmd_copy_buffer_to_image(
                    self.command_buffer,
                    allocation.staging_buffer,
                    allocation.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy_region(allocation)],
                );
                let to_general = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(full_color);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }

            // Upload every sampled texture and move it to SHADER_READ_ONLY,
            // the layout its SAMPLED_IMAGE descriptor promised.
            for allocation in &self.sampled {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: allocation.layers,
                };
                let to_transfer = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(range);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: allocation.layers,
                    })
                    .image_extent(vk::Extent3D {
                        width: allocation.width,
                        height: allocation.height,
                        depth: allocation.depth,
                    });
                self.device().cmd_copy_buffer_to_image(
                    self.command_buffer,
                    allocation.staging_buffer,
                    allocation.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                let to_sampled = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(range);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_sampled],
                );
            }

            self.device().cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            if !self.descriptor_set.is_null() {
                self.device().cmd_bind_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
            }
            if let Some(binding) = state.binding
                && !binding.push_constants.is_empty()
            {
                self.device().cmd_push_constants(
                    self.command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    binding.push_constant_offset,
                    &binding.push_constants,
                );
            }
            self.device().cmd_dispatch(
                self.command_buffer,
                state.groups[0],
                state.groups[1],
                state.groups[2],
            );

            // GDS contents persist across dispatches: make this dispatch's
            // GDS writes available to LATER dispatches' shader reads/writes
            // (a pipeline barrier's second scope covers subsequent
            // submissions on this queue; the fence alone only orders
            // device-to-host visibility).
            if !self.gds.is_null() {
                let gds_flush = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[gds_flush],
                    &[],
                    &[],
                );
            }

            // Copy every UAV back out for the guest-memory writeback.
            for allocation in &self.images {
                let to_readback = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(full_color);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_readback],
                );
                self.device().cmd_copy_image_to_buffer(
                    self.command_buffer,
                    allocation.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    allocation.readback_buffer,
                    &[copy_region(allocation)],
                );
            }
            if !self.images.is_empty() {
                // Make the transfer writes visible to the host map after the
                // fence wait.
                let host_read = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[host_read],
                    &[],
                    &[],
                );
            }
            self.device().end_command_buffer(self.command_buffer)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkEndCommandBuffer: {e}")))?;
        let command_buffers = [self.command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        // SAFETY: all submission handles are live; fence is unsignaled.
        unsafe {
            self.device()
                .queue_submit(self.dev.queue(), &submits, self.fence)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkQueueSubmit: {e}")))?;
        // SAFETY: waiting on this submission's live fence.
        unsafe { self.device().wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkWaitForFences: {e}")))
    }

    /// Map one host-coherent allocation and copy `size` bytes out.
    fn read_host_memory(&self, memory: vk::DeviceMemory, size: usize) -> Result<Vec<u8>, GpuError> {
        // SAFETY: host-coherent allocation is no longer used by the queue
        // (fence signaled) and is mapped for its initialized range.
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory: {e}")))?;
        // Fallible host allocation: a large compute readback under host memory
        // pressure must DEGRADE (return an error the dispatch path skips on),
        // never abort the whole process via the infallible allocator. Same
        // "degrade, not abort" policy as `draw_translate::alloc_zeroed`.
        let mut bytes: Vec<u8> = Vec::new();
        bytes.try_reserve_exact(size).map_err(|_| {
            // SAFETY: unmap the mapping we opened above before bailing.
            unsafe { self.device().unmap_memory(memory) };
            GpuError::VulkanInitFailed(format!(
                "compute readback: {size} B host allocation failed (out of memory) — \
                 skipping the dispatch instead of aborting"
            ))
        })?;
        bytes.resize(size, 0);
        // SAFETY: source mapped range and destination allocation both cover
        // `size` bytes and cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), bytes.as_mut_ptr(), size);
            self.device().unmap_memory(memory);
        }
        Ok(bytes)
    }

    fn read_storage(&self) -> Result<Vec<Vec<u8>>, GpuError> {
        self.storage
            .iter()
            .map(|allocation| self.read_host_memory(allocation.memory, allocation.size))
            .collect()
    }

    fn read_images(&self) -> Result<Vec<Vec<u8>>, GpuError> {
        self.images
            .iter()
            .map(|allocation| {
                let size = (allocation.width as usize)
                    * (allocation.height as usize)
                    * (allocation.depth as usize)
                    * (allocation.texel as usize);
                self.read_host_memory(allocation.readback_memory, size)
            })
            .collect()
    }
}

impl Drop for ComputeResources<'_> {
    fn drop(&mut self) {
        // NOT destroyed here — owned by the device's DrawCaches and reused by
        // the next draw/dispatch: fence, command buffer, pipeline, pipeline
        // layout, descriptor pool, descriptor layout, shader module, samplers.
        //
        // SAFETY: every handle destroyed below was created from this device
        // for this dispatch alone and is destroyed once, after synchronous
        // fence completion or a failed build.
        unsafe {
            while let Some(allocation) = self.storage.pop() {
                self.device().destroy_buffer(allocation.buffer, None);
                self.device().free_memory(allocation.memory, None);
            }
            if let Some(allocation) = self.eud_raw.take() {
                self.device().destroy_buffer(allocation.buffer, None);
                self.device().free_memory(allocation.memory, None);
            }
            self.samplers.clear();
            while let Some(allocation) = self.sampled.pop() {
                if !allocation.view.is_null() {
                    self.device().destroy_image_view(allocation.view, None);
                }
                if !allocation.image.is_null() {
                    self.device().destroy_image(allocation.image, None);
                }
                if !allocation.memory.is_null() {
                    self.device().free_memory(allocation.memory, None);
                }
                if !allocation.staging_buffer.is_null() {
                    self.device()
                        .destroy_buffer(allocation.staging_buffer, None);
                }
                if !allocation.staging_memory.is_null() {
                    self.device().free_memory(allocation.staging_memory, None);
                }
            }
            while let Some(allocation) = self.images.pop() {
                if !allocation.view.is_null() {
                    self.device().destroy_image_view(allocation.view, None);
                }
                if !allocation.image.is_null() {
                    self.device().destroy_image(allocation.image, None);
                }
                if !allocation.memory.is_null() {
                    self.device().free_memory(allocation.memory, None);
                }
                if !allocation.staging_buffer.is_null() {
                    self.device()
                        .destroy_buffer(allocation.staging_buffer, None);
                }
                if !allocation.staging_memory.is_null() {
                    self.device().free_memory(allocation.staging_memory, None);
                }
                if !allocation.readback_buffer.is_null() {
                    self.device()
                        .destroy_buffer(allocation.readback_buffer, None);
                }
                if !allocation.readback_memory.is_null() {
                    self.device().free_memory(allocation.readback_memory, None);
                }
            }
        }
    }
}
