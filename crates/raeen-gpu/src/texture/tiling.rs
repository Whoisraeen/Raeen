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

// ---------------------------------------------------------------------------
// GFX10 (RDNA2) exact swizzles — NOT the GCN approximations above.
// ---------------------------------------------------------------------------

/// One address bit of a GFX10 swizzle equation: the bit's value is the parity
/// of `(x & x_mask) ^ (y & y_mask)`. Mirrors SharpEmu `GnmTiling.cs`'s
/// `AddressBit`, itself a transcription of AMD AddrLib's
/// `gfx10SwizzlePattern.h` nibble tables (MIT).
#[derive(Clone, Copy)]
pub struct AddressBit {
    x_mask: u32,
    y_mask: u32,
}

const fn ab_x(bit: u8) -> AddressBit {
    AddressBit {
        x_mask: 1 << bit,
        y_mask: 0,
    }
}
const fn ab_y(bit: u8) -> AddressBit {
    AddressBit {
        x_mask: 0,
        y_mask: 1 << bit,
    }
}
const fn ab_xy(x_bit: u8, y_bit: u8) -> AddressBit {
    AddressBit {
        x_mask: 1 << x_bit,
        y_mask: 1 << y_bit,
    }
}
const fn ab_xyy(x_bit: u8, y1: u8, y2: u8) -> AddressBit {
    AddressBit {
        x_mask: 1 << x_bit,
        y_mask: (1 << y1) | (1 << y2),
    }
}
const AB_ZERO: AddressBit = AddressBit {
    x_mask: 0,
    y_mask: 0,
};

/// PS5/Oberon (16-pipe, 8-packer, "RB+") single-sample `SW_64KB_Z_X`
/// (SWIZZLE_MODE 24) equations — the DEPTH layout, which interleaves X and Y
/// from bit 0 instead of running X first as `_R_X` does. Transcribed from
/// SharpEmu's `RbPlus64KDepthX` (GnmTiling.cs, GPL-2.0), the same source and
/// topology as the two tables below. Measured need: ASTRO.BOT samples a
/// 1920x1080 format-22 (R32_SFLOAT) depth target with tile mode 24.
const RB_PLUS_64K_DEPTH_X: [[AddressBit; 16]; 5] = [
    // 1 byte/element: nibble01=8, nibble2=306, nibble3=379.
    [
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_x(2),
        ab_y(2),
        ab_x(3),
        ab_y(3),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_x(6),
        ab_y(6),
        ab_xy(7, 8),
        ab_xy(8, 7),
    ],
    // 2 bytes/element: nibble01=9, nibble2=306, nibble3=389.
    [
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_x(2),
        ab_y(2),
        ab_x(3),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(3),
        ab_x(6),
        ab_xy(7, 7),
        ab_xy(8, 6),
    ],
    // 4 bytes/element: nibble01=10, nibble2=306, nibble3=381.
    [
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_x(2),
        ab_y(2),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_x(3),
        ab_y(3),
        ab_xy(6, 7),
        ab_xy(7, 6),
    ],
    // 8 bytes/element: nibble01=11, nibble2=307, nibble3=382.
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_x(2),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(2),
        ab_x(3),
        ab_xy(7, 3),
        ab_xy(6, 6),
    ],
    // 16 bytes/element is identical to R_X for a 2D single-sample image.
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_x(2),
        ab_y(2),
        ab_xy(6, 3),
        ab_xy(3, 6),
    ],
];

/// PS5/Oberon (16-pipe, 8-packer, "RB+") single-sample `SW_64KB_R_X`
/// (SWIZZLE_MODE 27) equations, one row per bytes-per-element log2
/// (1/2/4/8/16-byte elements). Transcribed from SharpEmu's
/// `RbPlus64KRenderX`, verified there against AMD AddrLib's
/// `GFX10_SW_64K_R_X_1xaa_RBPLUS_PATINFO` rows.
const RB_PLUS_64K_RENDER_X: [[AddressBit; 16]; 5] = [
    // 1 byte/element: nibble01=0, nibble2=307, nibble3=379.
    [
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_x(3),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_y(3),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_x(6),
        ab_y(6),
        ab_xy(7, 8),
        ab_xy(8, 7),
    ],
    // 2 bytes/element: nibble01=1, nibble2=307, nibble3=389.
    [
        AB_ZERO,
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_x(3),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(3),
        ab_x(6),
        ab_xy(7, 7),
        ab_xy(8, 6),
    ],
    // 4 bytes/element: nibble01=39, nibble2=307, nibble3=381.
    [
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_x(1),
        ab_y(0),
        ab_y(1),
        ab_x(2),
        ab_y(2),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_x(3),
        ab_y(3),
        ab_xy(6, 7),
        ab_xy(7, 6),
    ],
    // 8 bytes/element: nibble01=6, nibble2=307, nibble3=382.
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_x(2),
        ab_y(1),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(2),
        ab_x(3),
        ab_xy(7, 3),
        ab_xy(6, 6),
    ],
    // 16 bytes/element: nibble01=7, nibble2=307, nibble3=390.
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_xyy(7, 4, 7),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_x(2),
        ab_y(2),
        ab_xy(6, 3),
        ab_xy(3, 6),
    ],
];

/// PS5/Oberon "RB+" `SW_64KB_S` (SWIZZLE_MODE 9) equations — the Standard
/// (non-XOR) 64 KiB layout. Transcribed from SharpEmu's `RbPlus64KStandard`
/// (AMD `GFX10_SW_64K_S_RBPLUS_PATINFO`).
const RB_PLUS_64K_STANDARD: [[AddressBit; 16]; 5] = [
    // 1 byte/element.
    [
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_x(3),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_y(3),
        ab_y(4),
        ab_x(4),
        ab_y(5),
        ab_x(5),
        ab_y(6),
        ab_x(6),
        ab_y(7),
        ab_x(7),
    ],
    // 2 bytes/element.
    [
        AB_ZERO,
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_x(3),
        ab_y(3),
        ab_x(4),
        ab_y(4),
        ab_x(5),
        ab_y(5),
        ab_x(6),
        ab_y(6),
        ab_x(7),
    ],
    // 4 bytes/element.
    [
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_x(1),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_x(2),
        ab_y(3),
        ab_x(3),
        ab_y(4),
        ab_x(4),
        ab_y(5),
        ab_x(5),
        ab_y(6),
        ab_x(6),
    ],
    // 8 bytes/element (also BC1/BC4 blocks).
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_y(1),
        ab_x(1),
        ab_x(2),
        ab_y(2),
        ab_x(3),
        ab_y(3),
        ab_x(4),
        ab_y(4),
        ab_x(5),
        ab_y(5),
        ab_x(6),
    ],
    // 16 bytes/element (also 16-byte BC blocks).
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_y(0),
        ab_y(1),
        ab_x(0),
        ab_x(1),
        ab_y(2),
        ab_x(2),
        ab_y(3),
        ab_x(3),
        ab_y(4),
        ab_x(4),
        ab_y(5),
        ab_x(5),
    ],
];

/// GFX10 `SW_4KB_S` (SWIZZLE_MODE 5) equations — the Standard (non-XOR) 4 KiB
/// layout, a SEPARATE 12-bit micro-tile equation from the 64 KiB Standard block
/// (using the larger equation leaves a regular grid in linearized atlases).
/// Transcribed from SharpEmu's `Standard4K` (`GnmTiling.cs`, AMD
/// `GFX10_SW_4K_S_PATINFO`). Each row is a 12-bit within-block byte offset (4
/// KiB = 2^12); the top 4 entries are `AB_ZERO` so the shared 16-bit
/// [`pattern_axis_term`] loop keeps the offset inside the 4 KiB block.
const STANDARD_4K: [[AddressBit; 16]; 5] = [
    // 1 byte/element.
    [
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_x(3),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_y(3),
        ab_y(4),
        ab_x(4),
        ab_y(5),
        ab_x(5),
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
    ],
    // 2 bytes/element.
    [
        AB_ZERO,
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_x(3),
        ab_y(3),
        ab_x(4),
        ab_y(4),
        ab_x(5),
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
    ],
    // 4 bytes/element.
    [
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_x(1),
        ab_y(0),
        ab_y(1),
        ab_y(2),
        ab_x(2),
        ab_y(3),
        ab_x(3),
        ab_y(4),
        ab_x(4),
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
    ],
    // 8 bytes/element (also BC1/BC4 blocks).
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_y(1),
        ab_x(1),
        ab_x(2),
        ab_y(2),
        ab_x(3),
        ab_y(3),
        ab_x(4),
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
    ],
    // 16 bytes/element (also 16-byte BC blocks).
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_y(0),
        ab_y(1),
        ab_x(0),
        ab_x(1),
        ab_y(2),
        ab_x(2),
        ab_y(3),
        ab_x(3),
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
    ],
];

/// The equation table + block byte size for a supported GFX10 swizzle mode, or
/// `None` for a mode with no ported equation yet (named at the call site). Mode
/// 5 is the 4 KiB Standard block; 9/24/27 are the 64 KiB Oberon RB+ blocks.
pub const fn swizzle_table(mode: u8) -> Option<(&'static [[AddressBit; 16]; 5], u64)> {
    match mode {
        5 => Some((&STANDARD_4K, 4096)),
        9 => Some((&RB_PLUS_64K_STANDARD, 65536)),
        24 => Some((&RB_PLUS_64K_DEPTH_X, 65536)),
        27 => Some((&RB_PLUS_64K_RENDER_X, 65536)),
        _ => None,
    }
}

/// The equation table for a supported 64 KiB swizzle mode, or `None` for a
/// mode with no ported equation yet (named at the call site instead).
pub const fn swizzle_64kb_table(mode: u8) -> Option<&'static [[AddressBit; 16]; 5]> {
    match mode {
        9 => Some(&RB_PLUS_64K_STANDARD),
        24 => Some(&RB_PLUS_64K_DEPTH_X),
        27 => Some(&RB_PLUS_64K_RENDER_X),
        _ => None,
    }
}

/// Byte offset of element (x, y) inside its swizzle block under an exact-XOR
/// equation. Coordinates are FULL-SURFACE: the bits above the block extent
/// (x7/y7 for a 128-px block) are the block-column/row parity the pipe/bank
/// XOR consumes — do not reduce them mod the block size.
///
/// The production detile hoists the two axis terms out of its loops (SharpEmu
/// #483); this whole-offset form is retained for the inverse writeback tiler
/// and the known-answer pins that check the factoring.
fn gfx10_pattern_offset(x: u32, y: u32, pattern: &[AddressBit; 16]) -> u64 {
    // Each output bit is parity(x & XMask) XOR parity(y & YMask), and parity
    // distributes over XOR, so the whole offset factors into two independent
    // axis terms: `x_term(x) ^ y_term(y)`. The direct form is kept for the
    // known-answer pins; the hot detile loop uses the factored terms below.
    pattern_axis_term(x, pattern, true) ^ pattern_axis_term(y, pattern, false)
}

/// One axis's contribution to a GFX10 swizzle offset: for each equation bit,
/// `parity(coord & mask) << bit`, where `mask` is the X or Y mask. Because
/// parity distributes over XOR, `offset(x, y) == axis(x, X) ^ axis(y, Y)`, so
/// the per-column X term can be precomputed once and the per-row Y term hoisted
/// out of the inner loop — one array load and one XOR per element instead of a
/// 16-bit interleave with 32 `count_ones` calls.
///
/// Ported from SharpEmu `GnmTiling.cs::PatternAxisTerm` (#483, commit 1f3963c,
/// GPL-2.0-or-later).
#[inline]
fn pattern_axis_term(coord: u32, pattern: &[AddressBit; 16], use_x: bool) -> u64 {
    let mut offset = 0u64;
    for (bit, eq) in pattern.iter().enumerate() {
        let mask = if use_x { eq.x_mask } else { eq.y_mask };
        let parity = (coord & mask).count_ones() & 1;
        offset |= u64::from(parity) << bit;
    }
    offset
}

/// Block extent in elements for a block of `block_bytes` at `bpp_log2`
/// bytes/element: the square-ish power-of-two grid (AddrLib
/// `ComputeThinBlockDimension` — width-precedent). 64 KiB at 4 B/el → 128x128.
fn block_dimensions(block_bytes: u32, bpp_log2: u32) -> (u32, u32) {
    let elements = block_bytes >> bpp_log2;
    let side_log2 = elements.trailing_zeros();
    let w_log2 = side_log2.div_ceil(2);
    (1 << w_log2, 1 << (side_log2 - w_log2))
}

/// Bytes a `SW_64KB_*` tiled surface occupies: whole blocks in each direction
/// (a surface smaller than a block still owns the whole block).
pub fn tiled_byte_count_64kb(width: u32, height: u32, bpp_log2: u32) -> u64 {
    const BLOCK_BYTES: u64 = 65536;
    let (bw, bh) = block_dimensions(BLOCK_BYTES as u32, bpp_log2);
    u64::from(width.div_ceil(bw)) * u64::from(height.div_ceil(bh)) * BLOCK_BYTES
}

/// Detile a `SW_64KB_R_X` (SWIZZLE_MODE 27) surface into tightly-packed
/// linear rows. `tiled` must cover the whole block grid (see
/// [`tiled_byte_count_64kb`]); pixels beyond it (never fetched) stay zero.
pub fn detile_64kb_r_x(tiled: &[u8], width: u32, height: u32, bpp_log2: u32) -> Vec<u8> {
    detile_64kb_with(tiled, width, height, bpp_log2, &RB_PLUS_64K_RENDER_X, 65536)
}

/// Detile a `SW_64KB_S` (SWIZZLE_MODE 9) surface — the Standard (non-XOR)
/// 64 KiB layout, measured on Minecraft's 1937x333 atlas texture.
pub fn detile_64kb_s(tiled: &[u8], width: u32, height: u32, bpp_log2: u32) -> Vec<u8> {
    detile_64kb_with(tiled, width, height, bpp_log2, &RB_PLUS_64K_STANDARD, 65536)
}

/// Detile a 64 KiB-block swizzled surface into tightly-packed linear rows
/// under `table`'s exact AddrLib equation. `tiled` must cover the whole
/// block grid; pixels beyond it (never fetched) stay zero.
fn detile_64kb_with(
    tiled: &[u8],
    width: u32,
    height: u32,
    bpp_log2: u32,
    table: &[[AddressBit; 16]; 5],
    block_bytes: u64,
) -> Vec<u8> {
    let pattern = &table[bpp_log2 as usize];
    let bpp = 1usize << bpp_log2;
    let (bw, bh) = block_dimensions(block_bytes as u32, bpp_log2);
    let blocks_per_row = u64::from(width.div_ceil(bw));
    let mut out = vec![0u8; width as usize * height as usize * bpp];
    // Precompute the per-column X term once (reused across every row): the
    // offset factors into `x_term ^ y_term`, so the inner loop drops from a
    // 16-bit interleave with 32 `count_ones` per element to one array load and
    // one XOR (SharpEmu #483 / 1f3963c). Detiling is per-texel work over
    // millions of elements, so this is the difference between a texture that
    // detiles in-frame and one that stalls the frame.
    let x_term_by_column: Vec<u64> = (0..width)
        .map(|xx| pattern_axis_term(xx, pattern, true))
        .collect();
    for yy in 0..height {
        let block_y = u64::from(yy / bh);
        let dest_row = yy as usize * width as usize * bpp;
        // The Y term is constant across the row; hoist it out of the inner loop.
        let y_term = pattern_axis_term(yy, pattern, false);
        for xx in 0..width {
            let block_index = block_y * blocks_per_row + u64::from(xx / bw);
            let src = block_index * block_bytes + (x_term_by_column[xx as usize] ^ y_term);
            let dst = dest_row + xx as usize * bpp;
            if src as usize + bpp <= tiled.len() {
                out[dst..dst + bpp].copy_from_slice(&tiled[src as usize..src as usize + bpp]);
            }
        }
    }
    out
}

/// Detile a supported GFX10 swizzle mode (4 KiB Standard 5, or 64 KiB 9/24/27),
/// or `None` for a mode with no ported equation yet (the caller names it instead
/// of guessing). The block size comes from [`swizzle_table`], so mode 5's 4 KiB
/// block and the 64 KiB modes share this one detiler.
pub fn detile_64kb(
    mode: u8,
    tiled: &[u8],
    width: u32,
    height: u32,
    bpp_log2: u32,
) -> Option<Vec<u8>> {
    let (table, block_bytes) = swizzle_table(mode)?;
    Some(detile_64kb_with(
        tiled,
        width,
        height,
        bpp_log2,
        table,
        block_bytes,
    ))
}

/// Bytes a supported GFX10 tiled surface occupies for `mode` — whole swizzle
/// blocks in each direction (a surface smaller than a block still owns the whole
/// block). `None` for an unsupported mode. Block size (4 KiB vs 64 KiB) comes
/// from [`swizzle_table`].
pub fn tiled_byte_count_for_mode(mode: u8, width: u32, height: u32, bpp_log2: u32) -> Option<u64> {
    let (_, block_bytes) = swizzle_table(mode)?;
    let (bw, bh) = block_dimensions(block_bytes as u32, bpp_log2);
    Some(u64::from(width.div_ceil(bw)) * u64::from(height.div_ceil(bh)) * block_bytes)
}

/// Tile a linear surface into a 64 KiB-block swizzle — the exact inverse of
/// [`detile_64kb_with`]. Production storage-image writeback uses
/// [`tile_64kb_into`]; this allocating wrapper is convenient for tests.
pub fn tile_64kb_r_x(linear: &[u8], width: u32, height: u32, bpp_log2: u32) -> Vec<u8> {
    tile_64kb_with(linear, width, height, bpp_log2, &RB_PLUS_64K_RENDER_X)
}

/// `SW_64KB_S` twin of [`tile_64kb_r_x`].
pub fn tile_64kb_s(linear: &[u8], width: u32, height: u32, bpp_log2: u32) -> Vec<u8> {
    tile_64kb_with(linear, width, height, bpp_log2, &RB_PLUS_64K_STANDARD)
}

/// `SW_64KB_Z_X` twin of [`tile_64kb_r_x`] (SWIZZLE_MODE 24).
pub fn tile_64kb_z_x(linear: &[u8], width: u32, height: u32, bpp_log2: u32) -> Vec<u8> {
    tile_64kb_with(linear, width, height, bpp_log2, &RB_PLUS_64K_DEPTH_X)
}

/// Tile a linear surface into a 64 KiB-block swizzle — the exact inverse of
/// [`detile_64kb_with`].
fn tile_64kb_with(
    linear: &[u8],
    width: u32,
    height: u32,
    bpp_log2: u32,
    table: &[[AddressBit; 16]; 5],
) -> Vec<u8> {
    const BLOCK_BYTES: u64 = 65536;
    let pattern = &table[bpp_log2 as usize];
    let bpp = 1usize << bpp_log2;
    let (bw, bh) = block_dimensions(BLOCK_BYTES as u32, bpp_log2);
    let blocks_per_row = u64::from(width.div_ceil(bw));
    let mut out = vec![0u8; tiled_byte_count_64kb(width, height, bpp_log2) as usize];
    for yy in 0..height {
        let block_y = u64::from(yy / bh);
        let src_row = yy as usize * width as usize * bpp;
        for xx in 0..width {
            let block_index = block_y * blocks_per_row + u64::from(xx / bw);
            let dst = block_index * BLOCK_BYTES + gfx10_pattern_offset(xx, yy, pattern);
            let src = src_row + xx as usize * bpp;
            out[dst as usize..dst as usize + bpp].copy_from_slice(&linear[src..src + bpp]);
        }
    }
    out
}

/// Retile one linear surface into an already-allocated guest swizzle buffer.
///
/// This is the non-allocating production inverse used after storage-image
/// readback. `false` means the mode is unsupported or either slice is shorter
/// than the exact linear/tiled extent.
pub fn tile_64kb_into(
    mode: u8,
    linear: &[u8],
    tiled: &mut [u8],
    width: u32,
    height: u32,
    bpp_log2: u32,
) -> bool {
    let Some((table, block_bytes)) = swizzle_table(mode) else {
        return false;
    };
    let bpp = 1usize << bpp_log2;
    let Some(linear_len) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(bpp))
    else {
        return false;
    };
    let Some(tiled_len) = tiled_byte_count_for_mode(mode, width, height, bpp_log2)
        .and_then(|bytes| usize::try_from(bytes).ok())
    else {
        return false;
    };
    if linear.len() < linear_len || tiled.len() < tiled_len {
        return false;
    }

    tiled[..tiled_len].fill(0);
    let pattern = &table[bpp_log2 as usize];
    let (bw, bh) = block_dimensions(block_bytes as u32, bpp_log2);
    let blocks_per_row = u64::from(width.div_ceil(bw));
    for yy in 0..height {
        let block_y = u64::from(yy / bh);
        let src_row = yy as usize * width as usize * bpp;
        for xx in 0..width {
            let block_index = block_y * blocks_per_row + u64::from(xx / bw);
            let dst = block_index * block_bytes + gfx10_pattern_offset(xx, yy, pattern);
            let src = src_row + xx as usize * bpp;
            tiled[dst as usize..dst as usize + bpp].copy_from_slice(&linear[src..src + bpp]);
        }
    }
    true
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
        assert!(
            seen.iter().all(|&s| s),
            "micro-tile order must be a bijection over 0..64"
        );
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

    /// Known-answer pins for the PS5/Oberon `SW_64KB_R_X` equation at 4 B/el
    /// (AddrLib `GFX10_SW_64K_R_X_1xaa_RBPLUS_PATINFO` nibble01=39).
    /// Element offsets within/between the 128x128 blocks.
    #[test]
    fn sw_64kb_r_x_equation_pins() {
        let p = &RB_PLUS_64K_RENDER_X[2]; // 4 bytes/element
        let at = |x: u32, y: u32| gfx10_pattern_offset(x, y, p);
        assert_eq!(at(0, 0), 0);
        assert_eq!(at(1, 0), 4, "x0 -> bit2");
        assert_eq!(at(2, 0), 8, "x1 -> bit3");
        assert_eq!(at(0, 1), 16, "y0 -> bit4");
        assert_eq!(at(0, 2), 32, "y1 -> bit5");
        assert_eq!(at(4, 0), 64, "x2 -> bit6");
        assert_eq!(at(0, 4), 128, "y2 -> bit7");
        assert_eq!(at(8, 0), 4096, "x3 -> bit12");
        assert_eq!(at(0, 8), 8192, "y3 -> bit13");
        assert_eq!(at(16, 0), 512, "x4 -> bit9 (x4^y4)");
        assert_eq!(at(0, 16), 768, "y4 -> bit8 + bit9");
        // The pipe/bank XOR: entering block column 2 (x=128, so x7=1) flips
        // bit8 (x7^y4^y7) AND bit15 (x7^y6); block row 2 (y=128, y7=1) flips
        // bit8 AND bit14 (x6^y7).
        assert_eq!(at(128, 0), 256 + 32768, "x7 -> bit8 + bit15");
        assert_eq!(at(0, 128), 256 + 16384, "y7 -> bit8 + bit14");
    }

    /// The factored `x_term ^ y_term` offset (SharpEmu #483) must stay
    /// byte-identical to the direct AddrLib address equation
    /// `parity(x & XMask) XOR parity(y & YMask)` per bit — over full-surface
    /// coordinates that engage the block-column/row XOR bits, for every ported
    /// table and every bytes-per-element row.
    #[test]
    fn factored_offset_matches_the_direct_address_equation() {
        fn direct(x: u32, y: u32, pattern: &[AddressBit; 16]) -> u64 {
            let mut offset = 0u64;
            for (bit, eq) in pattern.iter().enumerate() {
                let parity = ((x & eq.x_mask).count_ones() + (y & eq.y_mask).count_ones()) & 1;
                offset |= u64::from(parity) << bit;
            }
            offset
        }
        for mode in [9u8, 24, 27] {
            let table = swizzle_64kb_table(mode).expect("ported mode");
            for row in table {
                for y in 0..300u32 {
                    for x in 0..300u32 {
                        assert_eq!(
                            gfx10_pattern_offset(x, y, row),
                            direct(x, y, row),
                            "mode {mode} at ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sw_64kb_r_x_tile_then_detile_is_identity() {
        // 4 B/el, wider than one block so the block grid + XOR bits engage.
        let (w, h, bpp_log2) = (300u32, 100u32, 2u32);
        let bpp = 1usize << bpp_log2;
        let linear: Vec<u8> = (0..(w * h) as usize * bpp)
            .map(|i| (i % 251) as u8)
            .collect();
        let tiled = tile_64kb_r_x(&linear, w, h, bpp_log2);
        assert_eq!(
            tiled.len() as u64,
            tiled_byte_count_64kb(w, h, bpp_log2),
            "tiled footprint is whole blocks"
        );
        let back = detile_64kb_r_x(&tiled, w, h, bpp_log2);
        assert_eq!(back, linear, "detile ∘ tile must be the identity");
        assert_ne!(tiled[..linear.len()], linear[..], "swizzle must reorder");
    }

    /// SWIZZLE_MODE 24 (`SW_64KB_Z_X`, the depth layout) round-trips and is a
    /// DIFFERENT permutation from mode 27 — the depth equations interleave X
    /// and Y from bit 0 where `_R_X` runs X first, so a table transcription
    /// slip that duplicated `_R_X` would be caught here.
    #[test]
    fn sw_64kb_z_x_tile_then_detile_is_identity_and_differs_from_r_x() {
        let (w, h, bpp_log2) = (300u32, 100u32, 2u32);
        let bpp = 1usize << bpp_log2;
        let linear: Vec<u8> = (0..(w * h) as usize * bpp)
            .map(|i| (i % 251) as u8)
            .collect();

        let table = swizzle_64kb_table(24).expect("mode 24 is supported");
        let tiled = tile_64kb_with(&linear, w, h, bpp_log2, table);
        assert_eq!(
            tiled.len() as u64,
            tiled_byte_count_64kb(w, h, bpp_log2),
            "tiled footprint is whole blocks"
        );
        let back = detile_64kb(24, &tiled, w, h, bpp_log2).expect("mode 24 detiles");
        assert_eq!(back, linear, "detile ∘ tile must be the identity");
        assert_ne!(tiled[..linear.len()], linear[..], "swizzle must reorder");

        let r_x = tile_64kb_r_x(&linear, w, h, bpp_log2);
        assert_ne!(
            tiled, r_x,
            "depth (24) and render (27) swizzles must not be the same permutation"
        );
    }

    /// The 4 KiB Standard block (SWIZZLE_MODE 5) detiles as the exact inverse of
    /// its own tiler, and — crucially — is a DIFFERENT permutation from the
    /// 64 KiB Standard block (a transcription slip that reused the 64 KiB
    /// equation would be caught). Measured on ASTRO.BOT's 32x32 format-71 (8
    /// B/texel) tile-mode-5 texture.
    #[test]
    fn sw_4kb_s_tile_then_detile_is_identity_and_differs_from_64kb() {
        let (w, h, bpp_log2) = (32u32, 32u32, 3u32); // 8 B/texel
        let bpp = 1usize << bpp_log2;
        let linear: Vec<u8> = (0..(w * h) as usize * bpp)
            .map(|i| (i % 251) as u8)
            .collect();

        // Tile with the 4 KiB Standard equation (the inverse of the detiler).
        let pattern = &STANDARD_4K[bpp_log2 as usize];
        let (bw, bh) = block_dimensions(4096, bpp_log2);
        let blocks_per_row = u64::from(w.div_ceil(bw));
        let mut tiled =
            vec![0u8; tiled_byte_count_for_mode(5, w, h, bpp_log2).expect("mode 5") as usize];
        for y in 0..h {
            let block_y = u64::from(y / bh);
            let src_row = y as usize * w as usize * bpp;
            for x in 0..w {
                let block_index = block_y * blocks_per_row + u64::from(x / bw);
                let dst = (block_index * 4096 + gfx10_pattern_offset(x, y, pattern)) as usize;
                let src = src_row + x as usize * bpp;
                tiled[dst..dst + bpp].copy_from_slice(&linear[src..src + bpp]);
            }
        }

        let back = detile_64kb(5, &tiled, w, h, bpp_log2).expect("mode 5 detiles");
        assert_eq!(
            back, linear,
            "detile ∘ tile must be the identity for 4 KiB_S"
        );
        assert_ne!(tiled[..linear.len()], linear[..], "swizzle must reorder");

        // The 32x32x8B surface is two 4 KiB blocks; the 64 KiB Standard equation
        // would place every texel in one block with a different permutation.
        let pattern_64k = &RB_PLUS_64K_STANDARD[bpp_log2 as usize];
        let differs = (0..w * h).any(|i| {
            let (x, y) = (i % w, i / w);
            gfx10_pattern_offset(x, y, pattern) != gfx10_pattern_offset(x, y, pattern_64k)
        });
        assert!(differs, "4 KiB_S must not be the 64 KiB_S permutation");
    }

    #[test]
    fn sw_64kb_block_dimensions() {
        assert_eq!(block_dimensions(65536, 2), (128, 128), "64 KiB at 4 B/el");
        assert_eq!(block_dimensions(65536, 4), (64, 64), "64 KiB at 16 B/el");
        assert_eq!(
            tiled_byte_count_64kb(1920, 1080, 2),
            15 * 9 * 65536,
            "1080 rows pad to 9 blocks (measured Minecraft UI texture)"
        );
    }

    /// The Standard (non-XOR) 64 KiB layout at 4 B/el — a pure interleave,
    /// measured on Minecraft's 1937x333 atlas texture (SWIZZLE_MODE 9).
    #[test]
    fn sw_64kb_s_equation_pins() {
        let p = &RB_PLUS_64K_STANDARD[2]; // 4 bytes/element
        let at = |x: u32, y: u32| gfx10_pattern_offset(x, y, p);
        assert_eq!(at(0, 0), 0);
        assert_eq!(at(1, 0), 4, "x0 -> bit2");
        assert_eq!(at(2, 0), 8, "x1 -> bit3");
        assert_eq!(at(0, 1), 16, "y0 -> bit4");
        assert_eq!(at(0, 2), 32, "y1 -> bit5");
        assert_eq!(at(0, 4), 64, "y2 -> bit6");
        assert_eq!(at(4, 0), 128, "x2 -> bit7");
        assert_eq!(at(0, 8), 256, "y3 -> bit8");
        assert_eq!(at(8, 0), 512, "x3 -> bit9");
        // No XOR terms in the Standard layout: block column 2 is plain.
        assert_eq!(at(128, 0), 0, "x7 appears nowhere in the equation");
        assert_eq!(at(0, 128), 0, "y7 appears nowhere in the equation");
    }

    #[test]
    fn sw_64kb_s_tile_then_detile_is_identity() {
        let (w, h, bpp_log2) = (300u32, 100u32, 2u32);
        let bpp = 1usize << bpp_log2;
        let linear: Vec<u8> = (0..(w * h) as usize * bpp)
            .map(|i| (i % 251) as u8)
            .collect();
        let tiled = tile_64kb_s(&linear, w, h, bpp_log2);
        let back = detile_64kb_s(&tiled, w, h, bpp_log2);
        assert_eq!(back, linear, "detile ∘ tile must be the identity");
    }
}
