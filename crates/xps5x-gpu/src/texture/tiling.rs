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

/// Micro tiles are 8×8 pixel blocks.
const MICRO_TILE_DIM: u32 = 8;

/// Element index of pixel `(px, py)` *within* an 8×8 GCN "thin" micro tile.
///
/// The micro-tile interior is **not** row-major — it's a Z-order (Morton)
/// interleave of the low 3 bits of x and y: `x0 y0 x1 y1 x2 y2`. This is the
/// documented GCN thin micro-tile element equation (the previous code used a
/// plain `py*8 + px`, which no real GPU tiling format uses, so any nontrivial
/// texture came out scrambled).
///
/// Note: this pins the *thin* micro-tile order; the DEPTH/DISPLAY/ROTATED
/// micro-tile modes use different equations, and macro tiling adds
/// bank/pipe swizzling on top — hardware-exact validation across all modes
/// needs real texture dumps (tracked in the reference-port ledger).
#[inline]
fn micro_tile_element(px: u32, py: u32) -> u32 {
    (px & 1)
        | ((py & 1) << 1)
        | ((px & 2) << 1)
        | ((py & 2) << 2)
        | ((px & 4) << 2)
        | ((py & 4) << 3)
}

/// Detile a micro-tiled texture into linear (row-major) layout.
fn detile_micro(src: &[u8], width: u32, height: u32, bpp: u32) -> Vec<u8> {
    let row_pitch = width * bpp;
    let mut dst = vec![0u8; (row_pitch * height) as usize];
    let tiles_x = width.div_ceil(MICRO_TILE_DIM);
    let tile_bytes = MICRO_TILE_DIM * MICRO_TILE_DIM * bpp;
    let bpp_us = bpp as usize;

    for y in 0..height {
        for x in 0..width {
            let tile_index = (y / MICRO_TILE_DIM) * tiles_x + (x / MICRO_TILE_DIM);
            let elem = micro_tile_element(x % MICRO_TILE_DIM, y % MICRO_TILE_DIM);
            let src_offset = (tile_index * tile_bytes + elem * bpp) as usize;
            let dst_offset = ((y * row_pitch) + (x * bpp)) as usize;
            if src_offset + bpp_us <= src.len() && dst_offset + bpp_us <= dst.len() {
                dst[dst_offset..dst_offset + bpp_us]
                    .copy_from_slice(&src[src_offset..src_offset + bpp_us]);
            }
        }
    }
    dst
}

/// Tile a linear texture into micro-tiled layout — the exact inverse of
/// [`detile_micro`], so `detile_micro(tile_micro(x)) == x`. Used by the
/// round-trip consistency test and by any path that needs to hand tiled data
/// to guest-visible memory.
pub fn tile_micro(src: &[u8], width: u32, height: u32, bpp: u32) -> Vec<u8> {
    let row_pitch = width * bpp;
    let tiles_x = width.div_ceil(MICRO_TILE_DIM);
    let tiles_y = height.div_ceil(MICRO_TILE_DIM);
    let tile_bytes = MICRO_TILE_DIM * MICRO_TILE_DIM * bpp;
    let mut dst = vec![0u8; (tiles_x * tiles_y * tile_bytes) as usize];
    let bpp_us = bpp as usize;

    for y in 0..height {
        for x in 0..width {
            let tile_index = (y / MICRO_TILE_DIM) * tiles_x + (x / MICRO_TILE_DIM);
            let elem = micro_tile_element(x % MICRO_TILE_DIM, y % MICRO_TILE_DIM);
            let dst_offset = (tile_index * tile_bytes + elem * bpp) as usize;
            let src_offset = ((y * row_pitch) + (x * bpp)) as usize;
            if src_offset + bpp_us <= src.len() && dst_offset + bpp_us <= dst.len() {
                dst[dst_offset..dst_offset + bpp_us]
                    .copy_from_slice(&src[src_offset..src_offset + bpp_us]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_tile_element_is_the_documented_z_order() {
        // Interleave x0 y0 x1 y1 x2 y2 → element bits 0..6.
        assert_eq!(micro_tile_element(0, 0), 0);
        assert_eq!(micro_tile_element(1, 0), 1); // x0
        assert_eq!(micro_tile_element(0, 1), 2); // y0
        assert_eq!(micro_tile_element(1, 1), 3);
        assert_eq!(micro_tile_element(2, 0), 4); // x1
        assert_eq!(micro_tile_element(0, 2), 8); // y1
        assert_eq!(micro_tile_element(4, 0), 16); // x2
        assert_eq!(micro_tile_element(0, 4), 32); // y2
        assert_eq!(micro_tile_element(7, 7), 63); // all bits set
        // Every pixel in the 8×8 tile maps to a distinct element in 0..64.
        let mut seen = [false; 64];
        for y in 0..8 {
            for x in 0..8 {
                let e = micro_tile_element(x, y) as usize;
                assert!(!seen[e], "element {e} produced twice");
                seen[e] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "micro-tile order must be a bijection over 0..64");
    }

    #[test]
    fn tile_then_detile_is_identity() {
        // A 16×16, 4-bpp texture with distinct per-pixel bytes; tiling then
        // detiling must reproduce it exactly (round-trip consistency).
        let (w, h, bpp) = (16u32, 16u32, 4u32);
        let linear: Vec<u8> = (0..(w * h * bpp)).map(|i| (i % 251) as u8).collect();
        let tiled = tile_micro(&linear, w, h, bpp);
        let back = detile(&tiled, w, h, bpp, TilingMode::MicroTiled);
        assert_eq!(back, linear, "detile ∘ tile must be the identity");
        // Tiling actually reorders bytes (it isn't a no-op).
        assert_ne!(tiled, linear, "micro tiling must reorder the linear layout");
    }

    #[test]
    fn linear_mode_is_passthrough() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(detile(&data, 2, 1, 4, TilingMode::Linear), data);
    }
}
