#ifndef RAEEN_PLUGIN_V3_H
#define RAEEN_PLUGIN_V3_H

/*
 * Raeen vendor-neutral GPU present-plugin ABI v3.
 *
 * This header contains no vendor API and may be used by FSR, XeSS, DLSS, or
 * other Vulkan implementations. The authoritative Rust definitions live in
 * crates/raeen-gpu/src/present_plugin/cabi_v3.rs.
 *
 * A v3 binary also exports raeen_plugin_v2 for CPU-only/older hosts. Raeen
 * queries v3 requirements before creating Vulkan, then creates the v3 instance
 * only after every declared requirement has been enabled.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAEEN_PLUGIN_ABI_VERSION_V3 3u
#define RAEEN_V3_OK 0
#define RAEEN_V3_DECLINED 1
#define RAEEN_V3_BAD_INPUT (-1)

#define RAEEN_V3_MAX_INSTANCE_EXTENSIONS 16u
#define RAEEN_V3_MAX_DEVICE_EXTENSIONS 32u
#define RAEEN_V3_MAX_EXTENSION_NAME 256u

#define RAEEN_V3_QUEUE_GRAPHICS (1u << 0)
#define RAEEN_V3_QUEUE_COMPUTE (1u << 1)
#define RAEEN_V3_QUEUE_OPTICAL_FLOW (1u << 2)

#define RAEEN_V3_RESOURCE_BORROWED (1u << 0)
#define RAEEN_V3_RESOURCE_HOST_OWNS_LAYOUT (1u << 1)

#define RAEEN_V3_TEMPORAL_RESET (1u << 0)
#define RAEEN_V3_DEPTH_INVERTED (1u << 1)
#define RAEEN_V3_DEPTH_INFINITE (1u << 2)
#define RAEEN_V3_MOTION_VECTORS_DILATED (1u << 3)
#define RAEEN_V3_MOTION_VECTORS_JITTERED (1u << 4)
#define RAEEN_V3_ORTHOGRAPHIC (1u << 5)
#define RAEEN_V3_HAS_EXPOSURE_TEXTURE (1u << 6)

typedef struct RaeenExtensionNameV3 {
    uint32_t len;
    uint8_t bytes[RAEEN_V3_MAX_EXTENSION_NAME];
} RaeenExtensionNameV3;

typedef struct RaeenVulkanRequirementsV3 {
    uint32_t struct_size;
    uint32_t minimum_api_version;
    uint32_t instance_extension_count;
    uint32_t device_extension_count;
    RaeenExtensionNameV3
        instance_extensions[RAEEN_V3_MAX_INSTANCE_EXTENSIONS];
    RaeenExtensionNameV3 device_extensions[RAEEN_V3_MAX_DEVICE_EXTENSIONS];
    uint32_t required_queue_flags;
    uint32_t extra_graphics_queues;
    uint32_t extra_compute_queues;
    uint32_t extra_optical_flow_queues;
    uint64_t required_feature_flags;
    uint64_t reserved[7];
} RaeenVulkanRequirementsV3;

typedef struct RaeenVulkanHostV3 {
    uint64_t instance;
    uint64_t physical_device;
    uint64_t device;
    uint64_t graphics_queue;
    uint64_t compute_queue;
    uint64_t optical_flow_queue;
    uint32_t graphics_queue_family;
    uint32_t compute_queue_family;
    uint32_t optical_flow_queue_family;
    uint32_t reserved;
    const void *get_instance_proc_addr;
    const void *get_device_proc_addr;
} RaeenVulkanHostV3;

typedef struct RaeenRectV3 {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} RaeenRectV3;

/*
 * All resources are owned by Raeen and borrowed only for process(). A plugin
 * must not destroy or free any handle. device_memory may be zero when the host
 * deliberately withholds an implementation detail the pass does not require.
 */
typedef struct RaeenVulkanResourceV3 {
    uint64_t image;
    uint64_t image_view;
    uint64_t device_memory;
    uint32_t vk_format;
    uint32_t layout;
    uint32_t width;
    uint32_t height;
    uint32_t queue_family;
    uint32_t flags;
} RaeenVulkanResourceV3;

typedef struct RaeenFrameSyncV3 {
    uint64_t wait_semaphore;
    uint64_t wait_value;
    uint64_t signal_semaphore;
    uint64_t signal_value;
} RaeenFrameSyncV3;

/*
 * Vulkan clip-space conventions; matrices are column-major. Motion-vector
 * scale converts stored vector values to render-resolution pixels.
 */
typedef struct RaeenTemporalDataV3 {
    uint32_t flags;
    uint32_t reserved;
    float jitter_x;
    float jitter_y;
    float motion_vector_scale_x;
    float motion_vector_scale_y;
    float exposure_scale;
    float pre_exposure;
    float near_plane;
    float far_plane;
    float frame_time_ms;
    float camera_view_to_clip[16];
    float camera_clip_to_view[16];
    float camera_clip_to_previous_clip[16];
    float camera_previous_clip_to_clip[16];
} RaeenTemporalDataV3;

/*
 * command_buffer is recording and outside a render pass. The plugin records
 * commands but never submits the queue. output is allocated and owned by Raeen.
 */
typedef struct RaeenPresentFrameV3 {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t frame_index;
    uint64_t command_buffer;
    RaeenVulkanResourceV3 color;
    RaeenVulkanResourceV3 depth;
    RaeenVulkanResourceV3 motion_vectors;
    RaeenVulkanResourceV3 exposure;
    RaeenVulkanResourceV3 output;
    RaeenRectV3 render_rect;
    RaeenRectV3 output_rect;
    RaeenTemporalDataV3 temporal;
    RaeenFrameSyncV3 sync;
} RaeenPresentFrameV3;

typedef struct RaeenPluginOutputV3 {
    uint32_t struct_size;
    uint32_t output_layout;
    uint64_t reserved[4];
} RaeenPluginOutputV3;

typedef struct RaeenPluginV3 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t (*query_requirements)(RaeenVulkanRequirementsV3 *out);
    void *(*create)(const RaeenVulkanHostV3 *host);
    void (*destroy)(void *instance);
    size_t (*name)(void *instance, uint8_t *buffer, size_t capacity);
    uint32_t (*capabilities)(void *instance);
    int32_t (*process)(void *instance, const RaeenPresentFrameV3 *frame,
                       RaeenPluginOutputV3 *out);
    uintptr_t reserved[8];
} RaeenPluginV3;

const RaeenPluginV3 *raeen_plugin_v3(void);

#ifdef __cplusplus
}
#endif

#endif
