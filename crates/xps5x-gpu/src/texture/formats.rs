//! PS5 texture format definitions and Vulkan format mapping.

/// PS5 surface formats (GNM data formats).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Ps5TextureFormat {
    R8Unorm = 0x01,
    R8G8Unorm = 0x03,
    R8G8B8A8Unorm = 0x0A,
    R8G8B8A8Srgb = 0x0B,
    B8G8R8A8Unorm = 0x0C,
    B8G8R8A8Srgb = 0x0D,
    R16Float = 0x10,
    R16G16Float = 0x12,
    R16G16B16A16Float = 0x1A,
    R32Float = 0x20,
    R32G32Float = 0x22,
    R32G32B32A32Float = 0x2A,
    R10G10B10A2Unorm = 0x30,
    R11G11B10Float = 0x31,
    Bc1Unorm = 0x40,
    Bc1Srgb = 0x41,
    Bc2Unorm = 0x42,
    Bc2Srgb = 0x43,
    Bc3Unorm = 0x44,
    Bc3Srgb = 0x45,
    Bc4Unorm = 0x46,
    Bc4Snorm = 0x47,
    Bc5Unorm = 0x48,
    Bc5Snorm = 0x49,
    Bc7Unorm = 0x4C,
    Bc7Srgb = 0x4D,
    D32Float = 0x80,
    D16Unorm = 0x81,
    D32FloatS8Uint = 0x82,
    Unknown = 0xFF,
}

impl Ps5TextureFormat {
    /// Convert to Vulkan VkFormat value.
    pub fn to_vulkan_format(&self) -> u32 {
        match self {
            Self::R8Unorm => 9,            // VK_FORMAT_R8_UNORM
            Self::R8G8Unorm => 16,         // VK_FORMAT_R8G8_UNORM
            Self::R8G8B8A8Unorm => 37,     // VK_FORMAT_R8G8B8A8_UNORM
            Self::R8G8B8A8Srgb => 43,      // VK_FORMAT_R8G8B8A8_SRGB
            Self::B8G8R8A8Unorm => 44,     // VK_FORMAT_B8G8R8A8_UNORM
            Self::B8G8R8A8Srgb => 50,      // VK_FORMAT_B8G8R8A8_SRGB
            Self::R16Float => 76,          // VK_FORMAT_R16_SFLOAT
            Self::R16G16Float => 83,       // VK_FORMAT_R16G16_SFLOAT
            Self::R16G16B16A16Float => 97, // VK_FORMAT_R16G16B16A16_SFLOAT
            Self::R32Float => 100,         // VK_FORMAT_R32_SFLOAT
            Self::R32G32Float => 103,      // VK_FORMAT_R32G32_SFLOAT
            Self::R32G32B32A32Float => 109,// VK_FORMAT_R32G32B32A32_SFLOAT
            Self::Bc1Unorm => 131,         // VK_FORMAT_BC1_RGBA_UNORM_BLOCK
            Self::Bc1Srgb => 132,          // VK_FORMAT_BC1_RGBA_SRGB_BLOCK
            Self::Bc3Unorm => 137,         // VK_FORMAT_BC3_UNORM_BLOCK
            Self::Bc3Srgb => 138,          // VK_FORMAT_BC3_SRGB_BLOCK
            Self::Bc7Unorm => 145,         // VK_FORMAT_BC7_UNORM_BLOCK
            Self::Bc7Srgb => 146,          // VK_FORMAT_BC7_SRGB_BLOCK
            Self::D32Float => 126,         // VK_FORMAT_D32_SFLOAT
            Self::D16Unorm => 124,         // VK_FORMAT_D16_UNORM
            Self::D32FloatS8Uint => 130,   // VK_FORMAT_D32_SFLOAT_S8_UINT
            _ => 0, // VK_FORMAT_UNDEFINED
        }
    }
}
