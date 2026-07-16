//! One-shot Vulkan compute dispatch for translated guest shaders.
//!
//! Guest storage buffers are uploaded into host-visible coherent Vulkan
//! buffers, dispatched, then read back after the queue fence. The caller owns
//! copying those bytes back into identity-mapped guest memory.

use super::instance::VulkanDevice;
use super::offscreen::ShaderStageBinding;
use ash::vk::Handle;
use ash::{Device, vk};
use xps5x_core::error::GpuError;

pub struct ComputeState<'a> {
    pub groups: [u32; 3],
    pub spirv: &'a [u32],
    pub binding: Option<&'a ShaderStageBinding>,
}

pub fn dispatch_compute(
    dev: &VulkanDevice,
    state: &ComputeState<'_>,
) -> Result<Vec<Vec<u8>>, GpuError> {
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

    let mut resources = ComputeResources::new(dev);
    resources.build(state)?;
    resources.record_and_submit(state)?;
    resources.read_storage()
}

struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
}

struct ComputeResources<'a> {
    dev: &'a VulkanDevice,
    storage: Vec<BufferAllocation>,
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
    fn new(dev: &'a VulkanDevice) -> Self {
        Self {
            dev,
            storage: Vec::new(),
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
        let shader_info = vk::ShaderModuleCreateInfo::default().code(state.spirv);
        // SAFETY: SPIR-V is an aligned dword slice alive for the call.
        self.shader = unsafe { self.device().create_shader_module(&shader_info, None) }
            .map_err(|e| GpuError::ShaderCompilationFailed(format!("vkCreateShaderModule: {e}")))?;

        if let Some(binding) = state.binding
            && let Some(storage) = &binding.storage_buffers
        {
            if binding.descriptor_set_slot != 0 {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "compute descriptor set slot {} is not supported yet",
                    binding.descriptor_set_slot
                )));
            }
            if storage.buffers.is_empty() {
                return Err(GpuError::PipelineCreationFailed(
                    "compute storage descriptor array is empty".to_owned(),
                ));
            }
            for bytes in &storage.buffers {
                self.storage.push(self.create_buffer(bytes)?);
            }

            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(storage.binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(self.storage.len() as u32)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            // SAFETY: binding slice is live for the call.
            self.descriptor_layout = unsafe {
                self.device()
                    .create_descriptor_set_layout(&layout_info, None)
            }
            .map_err(|e| {
                GpuError::PipelineCreationFailed(format!("vkCreateDescriptorSetLayout: {e}"))
            })?;

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(self.storage.len() as u32)];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            // SAFETY: pool-size slice is live for the call.
            self.descriptor_pool =
                unsafe { self.device().create_descriptor_pool(&pool_info, None) }.map_err(|e| {
                    GpuError::PipelineCreationFailed(format!("vkCreateDescriptorPool: {e}"))
                })?;
            let layouts = [self.descriptor_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);
            // SAFETY: pool and layout are live handles from this device.
            self.descriptor_set = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }
                .map_err(|e| {
                    GpuError::PipelineCreationFailed(format!("vkAllocateDescriptorSets: {e}"))
                })?[0];
            let infos: Vec<_> = self
                .storage
                .iter()
                .map(|allocation| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(allocation.buffer)
                        .range(allocation.size as u64)
                })
                .collect();
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(storage.binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos)];
            // SAFETY: descriptor set and every named buffer are live.
            unsafe { self.device().update_descriptor_sets(&writes, &[]) };
        }

        let set_layouts: Vec<_> = (!self.descriptor_layout.is_null())
            .then_some(self.descriptor_layout)
            .into_iter()
            .collect();
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
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: layout/range slices and descriptor layout stay live.
        self.pipeline_layout = unsafe {
            self.device()
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|e| GpuError::PipelineCreationFailed(format!("vkCreatePipelineLayout: {e}")))?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout);
        // SAFETY: shader module and layout are live, and create info is local.
        self.pipeline = unsafe {
            self.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| {
            GpuError::PipelineCreationFailed(format!("vkCreateComputePipelines: {e}"))
        })?[0];

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.dev.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: command pool belongs to this live device.
        self.command_buffer = unsafe { self.device().allocate_command_buffers(&alloc_info) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("command buffer alloc: {e}")))?[0];
        // SAFETY: plain fence creation on a live device.
        self.fence = unsafe {
            self.device()
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateFence: {e}")))?;
        Ok(())
    }

    fn create_buffer(&self, bytes: &[u8]) -> Result<BufferAllocation, GpuError> {
        if bytes.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "zero-sized compute storage buffer".to_owned(),
            ));
        }
        let info = vk::BufferCreateInfo::default()
            .size(bytes.len() as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: create info is local and the device is live.
        let buffer = unsafe { self.device().create_buffer(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateBuffer: {e}")))?;
        // SAFETY: buffer is a live handle from this device.
        let req = unsafe { self.device().get_buffer_memory_requirements(buffer) };
        let memory_type = self.dev.find_memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type);
        // SAFETY: requirements and memory type belong to this buffer/device.
        let memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkAllocateMemory: {e}")))?;
        // SAFETY: buffer and allocation are compatible live handles.
        unsafe { self.device().bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkBindBufferMemory: {e}")))?;
        // SAFETY: host-visible allocation is mapped for the initialized range.
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory: {e}")))?;
        // SAFETY: mapped range is at least bytes.len(); pointers do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.device().unmap_memory(memory);
        }
        Ok(BufferAllocation {
            buffer,
            memory,
            size: bytes.len(),
        })
    }

    fn record_and_submit(&self, state: &ComputeState<'_>) -> Result<(), GpuError> {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: newly allocated command buffer is not pending.
        unsafe {
            self.device()
                .begin_command_buffer(self.command_buffer, &begin)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkBeginCommandBuffer: {e}")))?;
        // SAFETY: pipeline/layout/sets are live and command buffer is recording.
        unsafe {
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

    fn read_storage(&self) -> Result<Vec<Vec<u8>>, GpuError> {
        self.storage
            .iter()
            .map(|allocation| {
                // SAFETY: host-coherent allocation is no longer used by the
                // queue (fence signaled) and is mapped for its initialized range.
                let ptr = unsafe {
                    self.device().map_memory(
                        allocation.memory,
                        0,
                        allocation.size as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                }
                .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory: {e}")))?;
                let mut bytes = vec![0; allocation.size];
                // SAFETY: source mapped range and destination allocation both
                // cover allocation.size bytes and cannot overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ptr.cast::<u8>(),
                        bytes.as_mut_ptr(),
                        allocation.size,
                    );
                    self.device().unmap_memory(allocation.memory);
                }
                Ok(bytes)
            })
            .collect()
    }
}

impl Drop for ComputeResources<'_> {
    fn drop(&mut self) {
        // SAFETY: every non-null handle was created from this device and is
        // destroyed once, after synchronous fence completion or failed build.
        unsafe {
            if !self.fence.is_null() {
                self.device().destroy_fence(self.fence, None);
            }
            if !self.command_buffer.is_null() {
                self.device()
                    .free_command_buffers(self.dev.command_pool(), &[self.command_buffer]);
            }
            if !self.pipeline.is_null() {
                self.device().destroy_pipeline(self.pipeline, None);
            }
            if !self.pipeline_layout.is_null() {
                self.device()
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if !self.descriptor_pool.is_null() {
                self.device()
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if !self.descriptor_layout.is_null() {
                self.device()
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
            }
            if !self.shader.is_null() {
                self.device().destroy_shader_module(self.shader, None);
            }
            while let Some(allocation) = self.storage.pop() {
                self.device().destroy_buffer(allocation.buffer, None);
                self.device().free_memory(allocation.memory, None);
            }
        }
    }
}
