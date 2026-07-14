//! PS5 texture tiling mode conversion.
//!
//! AMD GPUs use tiled memory layouts for textures to optimize
//! cache locality. PS5 textures must be detiled before upload
//! to the host GPU (which may use different tiling).

/// PS5 tiling modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingMode {
    /// Linear layout (no tiling).
    Linear,
    /// Micro-tiled (1D tiling, 256-byte aligned rows).
    MicroTiled,
    /// Macro-tiled (2D tiling, bank-interleaved).
    MacroTiled,
}

/// Detile a texture from PS5 tiling to linear layout.
///
/// # Arguments
/// * `src` — Source texture data in tiled format
/// * `width` — Texture width in pixels
/// * `height` — Texture height in pixels
/// * `bpp` — Bytes per pixel
/// * `mode` — Source tiling mode
///
/// Returns the texture data in linear (row-major) layout.
pub fn detile(src: &[u8], width: u32, height: u32, bpp: u32, mode: TilingMode) -> Vec<u8> {
    match mode {
        TilingMode::Linear => src.to_vec(),
        TilingMode::MicroTiled => detile_micro(src, width, height, bpp),
        TilingMode::MacroTiled => detile_macro(src, width, height, bpp),
    }
}

/// Detile micro-tiled texture.
fn detile_micro(src: &[u8], width: u32, height: u32, bpp: u32) -> Vec<u8> {
    let row_pitch = width * bpp;
    let mut dst = vec![0u8; (row_pitch * height) as usize];

    // Micro tiles are 8x8 pixel blocks.
    let tile_width = 8u32;
    let tile_height = 8u32;
    let tiles_x = width.div_ceil(tile_width);
    let tiles_y = height.div_ceil(tile_height);
    let tile_bytes = tile_width * tile_height * bpp;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tile_index = ty * tiles_x + tx;
            let tile_offset = (tile_index * tile_bytes) as usize;

            for py in 0..tile_height {
                for px in 0..tile_width {
                    let x = tx * tile_width + px;
                    let y = ty * tile_height + py;

                    if x >= width || y >= height {
                        continue;
                    }

                    let src_offset = tile_offset + ((py * tile_width + px) * bpp) as usize;
                    let dst_offset = ((y * row_pitch) + (x * bpp)) as usize;

                    if src_offset + bpp as usize <= src.len()
                        && dst_offset + bpp as usize <= dst.len()
                    {
                        dst[dst_offset..dst_offset + bpp as usize]
                            .copy_from_slice(&src[src_offset..src_offset + bpp as usize]);
                    }
                }
            }
        }
    }

    dst
}

/// Detile macro-tiled texture (simplified).
fn detile_macro(src: &[u8], width: u32, height: u32, bpp: u32) -> Vec<u8> {
    // Macro tiling is complex (bank interleaving, pipe swizzling).
    // For now, fall back to micro detiling as an approximation.
    // A full implementation requires the exact tiling parameters
    // from the GPU's tiling register configuration.
    tracing::warn!("Macro detiling is approximate — visual artifacts may occur");
    detile_micro(src, width, height, bpp)
}
