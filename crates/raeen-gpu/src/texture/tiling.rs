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

/// PS5/Prospero single-sample `SW_64KB_R_X` (SWIZZLE_MODE 27) equations, one
/// row per bytes-per-element log2 (1/2/4/8/16-byte elements).
///
/// Behaviorally transcribed from KytyPS5
/// `guest_gpu/tile.cpp::Gen5RenderTargetOffsetInBlock`. This deliberately does
/// not use SharpEmu's Navi/RB+ pattern: a retail Minecraft capture whose T# is
/// mode 27 decodes coherently with the Prospero equation and is visibly
/// scrambled by that generic RB+ table. The high coordinate bits below are
/// the per-block XOR; callers must pass full-surface coordinates.
const PROSPERO_64K_RENDER_X: [[AddressBit; 16]; 5] = [
    // 1 byte/element.
    [
        ab_x(0),
        ab_x(1),
        ab_x(2),
        ab_y(1),
        ab_y(0),
        ab_y(2),
        ab_x(3),
        ab_y(4),
        ab_xy(3, 3),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
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
        ab_xy(3, 3),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(4),
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
        ab_xy(3, 3),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(3),
        ab_x(4),
        ab_y(6),
        ab_x(6),
    ],
    // 8 bytes/element.
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_x(2),
        ab_y(1),
        ab_xy(3, 3),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(2),
        ab_x(3),
        ab_y(4),
        ab_x(6),
    ],
    // 16 bytes/element.
    [
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        AB_ZERO,
        ab_x(0),
        ab_y(0),
        ab_x(1),
        ab_y(1),
        ab_xy(3, 3),
        ab_xy(4, 4),
        ab_xy(6, 5),
        ab_xy(5, 6),
        ab_y(2),
        ab_x(2),
        ab_y(3),
        ab_x(4),
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
        27 => Some((&PROSPERO_64K_RENDER_X, 65536)),
        _ => None,
    }
}

/// The equation table for a supported 64 KiB swizzle mode, or `None` for a
/// mode with no ported equation yet (named at the call site instead).
pub const fn swizzle_64kb_table(mode: u8) -> Option<&'static [[AddressBit; 16]; 5]> {
    match mode {
        9 => Some(&RB_PLUS_64K_STANDARD),
        24 => Some(&RB_PLUS_64K_DEPTH_X),
        27 => Some(&PROSPERO_64K_RENDER_X),
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
    detile_64kb_with(
        tiled,
        width,
        height,
        bpp_log2,
        &PROSPERO_64K_RENDER_X,
        65536,
    )
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

    // One output row per iteration, and rows never share destination bytes, so
    // the loop parallelizes exactly. Above the threshold the row work dominates
    // the split cost; below it, rayon's overhead would exceed the copy. Ported
    // from SharpEmu's `Parallel.For` over the same loop
    // (`reference/sharpemu/src/SharpEmu.Libs/Agc/GnmTiling.cs:533-539`), which
    // gates on the same order of element count.
    const PARALLEL_MIN_ELEMENTS: u64 = 512 * 512;
    let row_bytes = width as usize * bpp;
    let detile_row = |yy: u32, row: &mut [u8]| {
        let block_y = u64::from(yy / bh);
        // The Y term is constant across the row; hoist it out of the inner loop.
        let y_term = pattern_axis_term(yy, pattern, false);
        for xx in 0..width {
            let block_index = block_y * blocks_per_row + u64::from(xx / bw);
            let src = block_index * block_bytes + (x_term_by_column[xx as usize] ^ y_term);
            let dst = xx as usize * bpp;
            if src as usize + bpp <= tiled.len() {
                row[dst..dst + bpp].copy_from_slice(&tiled[src as usize..src as usize + bpp]);
            }
        }
    };

    if u64::from(width) * u64::from(height) >= PARALLEL_MIN_ELEMENTS {
        use rayon::prelude::*;
        out.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(yy, row)| detile_row(yy as u32, row));
    } else {
        for (yy, row) in out.chunks_mut(row_bytes).enumerate() {
            detile_row(yy as u32, row);
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
    if !bpp_log2_is_supported(bpp_log2) {
        return None;
    }
    Some(detile_64kb_with(
        tiled,
        width,
        height,
        bpp_log2,
        table,
        block_bytes,
    ))
}

/// Whether a bytes-per-element log2 has a row in the swizzle tables.
///
/// Callers derive this with `bytes.trailing_zeros()`, which is only a log2 for
/// a power of two and is silently wrong otherwise: a 3-byte (24-bit) element
/// yields 0 and would be detiled as 1 byte per texel, and a 32-byte element
/// yields 5 — one past the last table row, which would panic on the index.
/// The tables carry exactly the five RDNA2 element sizes (1/2/4/8/16 bytes), so
/// anything else is refused by name here and the caller reports an unsupported
/// texture instead of corrupting or crashing.
///
/// SharpEmu guards the same way with `BitLog2` returning -1 for a non-power-of-
/// two (`reference/sharpemu/src/SharpEmu.Libs/Agc/GnmTiling.cs`).
pub const fn bpp_log2_is_supported(bpp_log2: u32) -> bool {
    (bpp_log2 as usize) < SWIZZLE_TABLE_ROWS
}

/// Rows in every swizzle table: element sizes 1, 2, 4, 8, 16 bytes.
pub const SWIZZLE_TABLE_ROWS: usize = 5;

/// Bytes a supported GFX10 tiled surface occupies for `mode` — whole swizzle
/// blocks in each direction (a surface smaller than a block still owns the whole
/// block). `None` for an unsupported mode. Block size (4 KiB vs 64 KiB) comes
/// from [`swizzle_table`].
pub fn tiled_byte_count_for_mode(mode: u8, width: u32, height: u32, bpp_log2: u32) -> Option<u64> {
    let (_, block_bytes) = swizzle_table(mode)?;
    if !bpp_log2_is_supported(bpp_log2) {
        return None;
    }
    let (bw, bh) = block_dimensions(block_bytes as u32, bpp_log2);
    Some(u64::from(width.div_ceil(bw)) * u64::from(height.div_ceil(bh)) * block_bytes)
}

/// Tile a linear surface into a 64 KiB-block swizzle — the exact inverse of
/// [`detile_64kb_with`]. Production storage-image writeback uses
/// [`tile_64kb_into`]; this allocating wrapper is convenient for tests.
pub fn tile_64kb_r_x(linear: &[u8], width: u32, height: u32, bpp_log2: u32) -> Vec<u8> {
    tile_64kb_with(linear, width, height, bpp_log2, &PROSPERO_64K_RENDER_X)
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

// ---------------------------------------------------------------------------
// GFX10 mip chains: mip 0 is NOT at the descriptor base.
// ---------------------------------------------------------------------------

/// Block extent in ELEMENTS for a swizzle mode at `bpp_log2` bytes/element, or
/// `None` for a mode/element size with no ported equation. 64 KiB at 4 B/el →
/// 128x128; the 4 KiB Standard block at 4 B/el → 32x32.
///
/// Ported from SharpEmu `GnmTiling.TryGetBlockElementDimensions`
/// (`reference/sharpemu/src/SharpEmu.Libs/Agc/GnmTiling.cs`, GPL-2.0-or-later).
pub fn block_element_dimensions(mode: u8, bpp_log2: u32) -> Option<(u32, u32)> {
    let (_, block_bytes) = swizzle_table(mode)?;
    if !bpp_log2_is_supported(bpp_log2) {
        return None;
    }
    let dims = block_dimensions(block_bytes as u32, bpp_log2);
    (dims.0 != 0 && dims.1 != 0).then_some(dims)
}

/// Where mip 0 of a GFX10 mip chain sits relative to the T#'s descriptor base.
///
/// AddrLib stores a GFX10 chain **smallest-first**
/// (`Gfx10Lib::ComputeSurfaceInfoMacroTiled`): the small levels pack together
/// into the *mip tail* occupying the FIRST swizzle block, the remaining levels
/// follow in DECREASING size, and **mip 0 lands at the end of the allocation**.
/// Reading a mipped surface at the descriptor base therefore decodes the mip
/// tail as if it were a full-extent mip 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseMipPlacement {
    /// Bytes from the descriptor base to mip 0's own block grid. Zero in the
    /// [`Self::tail_element`] case (mip 0 lives inside the first block).
    pub byte_offset: u64,
    /// Bytes one array slice's WHOLE chain occupies — the array-layer stride,
    /// which for a mipped surface is larger than mip 0's own block grid.
    pub chain_slice_bytes: u64,
    /// `Some((x, y))` when the entire chain fits inside the mip tail block: mip
    /// 0 is then the sub-rectangle at that element coordinate of the *detiled*
    /// block, not a block grid of its own.
    pub tail_element: Option<(u32, u32)>,
}

/// Locate mip 0 in a GFX10 mip chain. `None` means "no relocation to apply":
/// a single-level resource, an unsupported swizzle mode/element size, or a
/// tail sub-rectangle that failed its bounds check (the caller then keeps the
/// descriptor base, i.e. today's behaviour, and says so).
///
/// `resource_mip_levels` is the ALLOCATION's level count (`MAX_MIP + 1`), not a
/// view's `BASE_LEVEL..=LAST_LEVEL` range.
///
/// Ported from SharpEmu `GnmTiling.TryGetBaseMipPlacement` (#470, commit
/// 6ee445f, GPL-2.0-or-later).
pub fn base_mip_placement(
    mode: u8,
    elements_wide: u32,
    elements_high: u32,
    bpp_log2: u32,
    resource_mip_levels: u32,
) -> Option<BaseMipPlacement> {
    if resource_mip_levels <= 1 || elements_wide == 0 || elements_high == 0 {
        return None;
    }
    let (_, block_bytes) = swizzle_table(mode)?;
    if !bpp_log2_is_supported(bpp_log2) {
        return None;
    }
    let (block_width, block_height) = block_dimensions(block_bytes as u32, bpp_log2);
    let block_size_log2 = block_bytes.trailing_zeros();
    if block_width == 0 || block_height == 0 || block_size_log2 < 8 {
        return None;
    }

    // AddrLib caps a chain at 16 levels; `mip_sizes` is sized to match.
    let mip_levels = resource_mip_levels.min(16);
    // Levels the tail block can absorb (AddrLib `GetMipTailInfo`): a 256 B block
    // has no tail, 512 B..2 KiB blocks scale with the block, and 4 KiB/64 KiB
    // blocks take `log2(block) - 4` (8 and 12 levels).
    let max_mips_in_tail = if block_size_log2 <= 8 {
        0
    } else if block_size_log2 <= 11 {
        1 + (1u32 << (block_size_log2 - 9))
    } else {
        block_size_log2 - 4
    };
    // The tail starts at half a block: an odd `log2(block)` splits the extra bit
    // into X, an even one into Y.
    let (tail_width, tail_height) = if block_size_log2 & 1 != 0 {
        (block_width >> 1, block_height)
    } else {
        (block_width, block_height >> 1)
    };

    let mut first_mip_in_tail = mip_levels;
    let mut mip_sizes = [0u64; 16];
    for i in 0..mip_levels {
        let mip_width = (elements_wide >> i).max(1);
        let mip_height = (elements_high >> i).max(1);
        if max_mips_in_tail > 0
            && mip_width <= tail_width
            && mip_height <= tail_height
            && mip_levels - i <= max_mips_in_tail
        {
            first_mip_in_tail = i;
            break;
        }
        // Every non-tail level owns whole blocks in both directions.
        let aligned_width = u64::from(mip_width.div_ceil(block_width) * block_width);
        let aligned_height = u64::from(mip_height.div_ceil(block_height) * block_height);
        mip_sizes[i as usize] = aligned_width * aligned_height * (1u64 << bpp_log2);
    }

    if first_mip_in_tail == 0 {
        // The whole chain — mip 0 included — lives in the tail block. Mip 0 is a
        // sub-rectangle of the detiled block at the tail slot's micro-block
        // coordinate, recovered from the slot's Morton-scattered offset.
        let (tail_x, tail_y) = mip_tail_element(max_mips_in_tail, block_size_log2, bpp_log2)?;
        if tail_x + elements_wide > block_width || tail_y + elements_high > block_height {
            return None;
        }
        return Some(BaseMipPlacement {
            byte_offset: 0,
            chain_slice_bytes: block_bytes,
            tail_element: Some((tail_x, tail_y)),
        });
    }

    // Smallest-first: [tail block][mip firstMipInTail-1] … [mip 1][mip 0].
    let mut byte_offset = if first_mip_in_tail < mip_levels {
        block_bytes
    } else {
        0
    };
    let mut chain_slice_bytes = byte_offset;
    for i in (1..first_mip_in_tail).rev() {
        byte_offset += mip_sizes[i as usize];
    }
    for i in 0..first_mip_in_tail {
        chain_slice_bytes += mip_sizes[i as usize];
    }
    Some(BaseMipPlacement {
        byte_offset,
        chain_slice_bytes,
        tail_element: None,
    })
}

/// Element coordinate of the last tail slot inside the mip tail block.
///
/// AddrLib's tail slot offsets are `m << 8` for the first seven slots and
/// `16 << m` beyond, and the slot's micro-block coordinate is that offset's
/// Morton de-interleave above bit 8 (odd bits → X, even bits → Y).
fn mip_tail_element(
    max_mips_in_tail: u32,
    block_size_log2: u32,
    bpp_log2: u32,
) -> Option<(u32, u32)> {
    let m = max_mips_in_tail.checked_sub(1)?;
    let mip_offset: u32 = if m > 6 { 16u32 << m } else { m << 8 };
    let mut mip_x = ((mip_offset >> 9) & 1)
        | ((mip_offset >> 10) & 2)
        | ((mip_offset >> 11) & 4)
        | ((mip_offset >> 12) & 8)
        | ((mip_offset >> 13) & 16)
        | ((mip_offset >> 14) & 32);
    let mut mip_y = ((mip_offset >> 8) & 1)
        | ((mip_offset >> 9) & 2)
        | ((mip_offset >> 10) & 4)
        | ((mip_offset >> 11) & 8)
        | ((mip_offset >> 12) & 16)
        | ((mip_offset >> 13) & 32);
    if block_size_log2 & 1 != 0 {
        std::mem::swap(&mut mip_x, &mut mip_y);
        if bpp_log2 & 1 != 0 {
            mip_y = (mip_y << 1) | (mip_x & 1);
            mip_x >>= 1;
        }
    }
    let (micro_width, micro_height) = block_dimensions(256, bpp_log2);
    if micro_width == 0 || micro_height == 0 {
        return None;
    }
    Some((mip_x * micro_width, mip_y * micro_height))
}

/// Detile the mip tail block and lift mip 0's sub-rectangle out of it, for the
/// [`BaseMipPlacement::tail_element`] case. `tiled` must cover one whole
/// swizzle block. `None` for an unsupported mode/element size, a short input,
/// or a sub-rectangle that does not fit the block.
pub fn detile_mip_tail_base(
    mode: u8,
    tiled: &[u8],
    elements_wide: u32,
    elements_high: u32,
    bpp_log2: u32,
    tail_element_x: u32,
    tail_element_y: u32,
) -> Option<Vec<u8>> {
    let (_, block_bytes) = swizzle_table(mode)?;
    let (block_width, block_height) = block_element_dimensions(mode, bpp_log2)?;
    if tiled.len() < usize::try_from(block_bytes).ok()?
        || tail_element_x + elements_wide > block_width
        || tail_element_y + elements_high > block_height
    {
        return None;
    }
    // Deswizzle the FULL block: mip 0's rows are interleaved with the other tail
    // levels in the tiled bytes, so they only become contiguous once linear.
    let block = detile_64kb(mode, tiled, block_width, block_height, bpp_log2)?;
    let bpp = 1usize << bpp_log2;
    let row = elements_wide as usize * bpp;
    let mut out = vec![0u8; row * elements_high as usize];
    for y in 0..elements_high as usize {
        let src =
            ((tail_element_y as usize + y) * block_width as usize + tail_element_x as usize) * bpp;
        out[y * row..(y + 1) * row].copy_from_slice(&block[src..src + row]);
    }
    Some(out)
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

    /// Callers derive `bpp_log2` with `trailing_zeros()`, which is a log2 only
    /// for a power of two. A 3-byte element yields 0 (would silently detile as
    /// 1 byte per texel) and a 32-byte element yields 5 — one past the last
    /// table row, which would have panicked on the index. Both must be refused.
    #[test]
    fn unsupported_element_sizes_are_refused_not_guessed() {
        assert_eq!(SWIZZLE_TABLE_ROWS, 5, "1, 2, 4, 8, 16 bytes per element");
        for supported in 0..SWIZZLE_TABLE_ROWS as u32 {
            assert!(bpp_log2_is_supported(supported));
            assert!(tiled_byte_count_for_mode(27, 64, 64, supported).is_some());
        }
        for bad in [SWIZZLE_TABLE_ROWS as u32, 6, 31] {
            assert!(!bpp_log2_is_supported(bad));
            assert!(
                tiled_byte_count_for_mode(27, 64, 64, bad).is_none(),
                "element size log2 {bad} has no table row"
            );
            assert!(
                detile_64kb(27, &[0u8; 4096], 64, 64, bad).is_none(),
                "detile must refuse element size log2 {bad} rather than panic"
            );
        }
        // An unsupported MODE still refuses regardless of element size.
        assert!(tiled_byte_count_for_mode(4, 64, 64, 2).is_none());
    }

    /// The row-parallel detile must produce byte-identical output to the serial
    /// path. Exercised at a size ABOVE the parallel threshold (512*512
    /// elements) so the rayon branch is the one under test, and compared
    /// against a tile/detile round trip.
    #[test]
    fn parallel_detile_matches_the_serial_result() {
        // 1024x512 @ 4 B/el = 524_288 elements, over the threshold.
        let (width, height, bpp_log2) = (1024u32, 512u32, 2u32);
        let bpp = 1usize << bpp_log2;
        let linear: Vec<u8> = (0..width as usize * height as usize * bpp)
            .map(|i| ((i * 2654435761usize) >> 7) as u8)
            .collect();

        let tiled = tile_64kb_r_x(&linear, width, height, bpp_log2);
        let round_tripped =
            detile_64kb(27, &tiled, width, height, bpp_log2).expect("mode 27 is supported");
        assert_eq!(
            round_tripped, linear,
            "parallel detile must invert the tiler exactly"
        );

        // And the same surface below the threshold takes the serial branch.
        let (sw, sh) = (64u32, 64u32);
        let small: Vec<u8> = (0..sw as usize * sh as usize * bpp)
            .map(|i| i as u8)
            .collect();
        let small_tiled = tile_64kb_r_x(&small, sw, sh, bpp_log2);
        assert_eq!(
            detile_64kb(27, &small_tiled, sw, sh, bpp_log2).expect("mode 27"),
            small,
            "serial branch must agree with the parallel one"
        );
    }

    /// Known-answer pins for the PS5/Prospero `SW_64KB_R_X` equation at 4 B/el.
    /// These are independently readable from KytyPS5's shift/mask equation.
    #[test]
    fn sw_64kb_r_x_equation_pins() {
        let p = &PROSPERO_64K_RENDER_X[2]; // 4 bytes/element
        let at = |x: u32, y: u32| gfx10_pattern_offset(x, y, p);
        assert_eq!(at(0, 0), 0);
        assert_eq!(at(1, 0), 4, "x0 -> bit2");
        assert_eq!(at(2, 0), 8, "x1 -> bit3");
        assert_eq!(at(0, 1), 16, "y0 -> bit4");
        assert_eq!(at(0, 2), 32, "y1 -> bit5");
        assert_eq!(at(0, 4), 64, "y2 -> bit6");
        assert_eq!(at(4, 0), 128, "x2 -> bit7");
        assert_eq!(at(8, 0), 256, "x3 -> bit8");
        assert_eq!(at(0, 8), 256 + 4096, "y3 -> bit8 + bit12");
        assert_eq!(at(16, 0), 512 + 8192, "x4 -> bit9 + bit13");
        assert_eq!(at(0, 16), 512, "y4 -> bit9");
        assert_eq!(at(64, 0), 1024 + 32768, "x6 -> bit10 + bit15");
        assert_eq!(at(0, 64), 2048 + 16384, "y6 -> bit11 + bit14");
    }

    /// Cross-check every Prospero row against KytyPS5's independently written
    /// shift/mask equations. This catches a transcription error that a
    /// tile-then-detile round trip cannot, because that round trip shares one
    /// table in both directions.
    #[test]
    fn sw_64kb_r_x_matches_kytyps5_prospero_equations() {
        fn reference(x: u32, y: u32, bpp_log2: u32) -> u64 {
            let mut offset = 0u32;
            match bpp_log2 {
                0 => {
                    offset ^= (y << 2) & 0x0008;
                    offset ^= (y << 4) & 0x0010;
                    offset ^= (y << 3) & 0x00a0;
                    offset ^= (y << 5) & 0x0f00;
                    offset ^= (y << 6) & 0x1000;
                    offset ^= (y << 7) & 0x4000;
                    offset ^= x & 0x0007;
                    offset ^= (x << 3) & 0x0040;
                    offset ^= (x << 5) & 0x0300;
                    offset ^= (x << 4) & 0x0400;
                    offset ^= (x << 6) & 0x0800;
                    offset ^= (x << 7) & 0x2000;
                    offset ^= (x << 8) & 0x8000;
                }
                1 => {
                    offset ^= (y << 4) & 0x0070;
                    offset ^= (y << 5) & 0x0f00;
                    offset ^= (y << 8) & 0x5000;
                    offset ^= (x << 1) & 0x000e;
                    offset ^= (x << 4) & 0x0480;
                    offset ^= (x << 5) & 0x0300;
                    offset ^= (x << 6) & 0x0800;
                    offset ^= (x << 7) & 0x2000;
                    offset ^= (x << 8) & 0x8000;
                }
                2 => {
                    offset ^= (y << 4) & 0x0070;
                    offset ^= (y << 5) & 0x0f00;
                    offset ^= (y << 9) & 0x1000;
                    offset ^= (y << 8) & 0x4000;
                    offset ^= (x << 2) & 0x000c;
                    offset ^= (x << 5) & 0x0380;
                    offset ^= (x << 4) & 0x0400;
                    offset ^= (x << 6) & 0x0800;
                    offset ^= (x << 9) & 0xa000;
                }
                3 => {
                    offset ^= (y << 4) & 0x0010;
                    offset ^= (y << 6) & 0x0080;
                    offset ^= (y << 5) & 0x0f00;
                    offset ^= (y << 10) & 0x5000;
                    offset ^= (x << 3) & 0x0008;
                    offset ^= (x << 4) & 0x0460;
                    offset ^= (x << 5) & 0x0300;
                    offset ^= (x << 6) & 0x0800;
                    offset ^= (x << 10) & 0x2000;
                    offset ^= (x << 9) & 0x8000;
                }
                4 => {
                    offset ^= (x << 4) & 0x0410;
                    offset ^= (x << 5) & 0x0340;
                    offset ^= (x << 6) & 0x0800;
                    offset ^= (x << 11) & 0xa000;
                    offset ^= (y << 5) & 0x0f20;
                    offset ^= (y << 6) & 0x0080;
                    offset ^= (y << 10) & 0x1000;
                    offset ^= (y << 11) & 0x4000;
                }
                _ => unreachable!(),
            }
            u64::from(offset)
        }

        for (bpp_log2, pattern) in PROSPERO_64K_RENDER_X.iter().enumerate() {
            for y in 0..300 {
                for x in 0..300 {
                    assert_eq!(
                        gfx10_pattern_offset(x, y, pattern),
                        reference(x, y, bpp_log2 as u32),
                        "{} B/el at ({x}, {y})",
                        1 << bpp_log2
                    );
                }
            }
        }
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

    /// The 2 B/element mode-27 row checked against an INDEPENDENT Prospero
    /// equation — not this file's own tables.
    ///
    /// The round-trip tests above tile with the inverse of the same table, so
    /// they are self-consistent even if a table row were transcribed wrong; and
    /// the known-answer pins cover the 4 B/element row only. This mask table is
    /// independently expanded from KytyPS5
    /// `guest_gpu/tile.cpp::Gen5RenderTargetOffsetInBlock<uint16_t>` (GPL-2.0
    /// with Kyty/MIT lineage). A tiled buffer laid out by this table must detile
    /// into ascending element indices byte-for-byte.
    #[test]
    fn sw_64kb_r_x_2bpp_matches_an_independent_re_derivation() {
        // (x_mask, y_mask) per output bit, Prospero 64 KiB R_X at 2 B/element.
        const REFERENCE: [(u32, u32); 16] = [
            (0, 0),
            (1 << 0, 0),
            (1 << 1, 0),
            (1 << 2, 0),
            (0, 1 << 0),
            (0, 1 << 1),
            (0, 1 << 2),
            (1 << 3, 0),
            (1 << 3, 1 << 3),
            (1 << 4, 1 << 4),
            (1 << 6, 1 << 5),
            (1 << 5, 1 << 6),
            (0, 1 << 4),
            (1 << 6, 0),
            (0, 1 << 6),
            (1 << 7, 0),
        ];
        fn reference_offset(x: u32, y: u32) -> u64 {
            let mut offset = 0u64;
            for (bit, (x_mask, y_mask)) in REFERENCE.iter().enumerate() {
                let parity = ((x & x_mask).count_ones() + (y & y_mask).count_ones()) & 1;
                offset |= u64::from(parity) << bit;
            }
            offset
        }

        const BLOCK_BYTES: u64 = 65536;
        let bpp_log2 = 1u32; // 2 bytes/element
        // 32768 elements/block: 15 bits split 8/7, x favored → 256x128. The
        // independent derivation assumes this split; a disagreement here would
        // invalidate the layout below, so pin it first.
        const BLOCK_W: u32 = 256;
        const BLOCK_H: u32 = 128;
        assert_eq!(
            block_dimensions(BLOCK_BYTES as u32, bpp_log2),
            (BLOCK_W, BLOCK_H)
        );

        // 384x200 exercises partial blocks; 768x512 exercises a 3x4 block grid
        // (and the u16 element index deliberately wraps — same on both sides).
        for (w, h) in [(384u32, 200u32), (768, 512)] {
            let blocks_per_row = u64::from(w.div_ceil(BLOCK_W));
            let tiled_len = tiled_byte_count_for_mode(27, w, h, bpp_log2)
                .expect("mode 27 at 2 B/el is supported");
            let mut tiled = vec![0u8; tiled_len as usize];
            for y in 0..h {
                for x in 0..w {
                    let block_index =
                        u64::from(y / BLOCK_H) * blocks_per_row + u64::from(x / BLOCK_W);
                    let src = (block_index * BLOCK_BYTES + reference_offset(x, y)) as usize;
                    let index = (y * w + x) as u16;
                    tiled[src] = index as u8;
                    tiled[src + 1] = (index >> 8) as u8;
                }
            }

            let linear = detile_64kb(27, &tiled, w, h, bpp_log2).expect("mode 27 detiles");
            assert_eq!(linear.len(), (w * h * 2) as usize);
            for i in 0..(w * h) as usize {
                let value = u16::from_le_bytes([linear[i * 2], linear[i * 2 + 1]]);
                assert_eq!(
                    value, i as u16,
                    "element {i} of {w}x{h} must come back in ascending order"
                );
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

    /// A GFX10 chain is stored smallest-first, so mip 0 is at the END of the
    /// allocation — the whole point of [`base_mip_placement`]. Worked by hand for
    /// a 512x512 RGBA8 `SW_64KB_S` surface with 10 levels:
    ///
    /// * block = 65536 B → 128x128 elements, `log2(block)` = 16
    /// * mips in tail = 16 - 4 = 12; tail extent = 128x64
    /// * mip 0 (512x512) and mip 1 (256x256) are too tall for the tail; mip 2
    ///   (128x128) is too tall as well; mip 3 (64x64) fits → the tail starts at 3
    /// * layout = [tail 64 KiB][mip 2 = 64 KiB][mip 1 = 256 KiB][mip 0 = 1 MiB]
    /// * so mip 0 begins 65536 + 65536 + 262144 = 393216 B (0x60000) past base
    #[test]
    fn base_mip_placement_puts_mip_zero_at_the_end_of_the_chain() {
        let placement = base_mip_placement(9, 512, 512, 2, 10).expect("a 10-level chain places");
        assert_eq!(placement.byte_offset, 393_216, "0x60000 past the T# base");
        assert_eq!(placement.chain_slice_bytes, 1_441_792);
        assert_eq!(placement.tail_element, None, "mip 0 has its own block grid");
        // Mip 0 really is LAST: its own block grid ends exactly at the slice end.
        assert_eq!(
            placement.byte_offset + tiled_byte_count_for_mode(9, 512, 512, 2).unwrap(),
            placement.chain_slice_bytes,
            "offset + mip 0's block grid must be the whole slice"
        );
    }

    /// The same math on the 4 KiB Standard block (mode 5), whose tail absorbs 8
    /// levels of a 32x16-element tail extent: 256x256 RGBA8 with 9 levels packs
    /// [tail 4 KiB][mip 3 = 4 KiB][mip 2 = 16 KiB][mip 1 = 64 KiB][mip 0 = 256 KiB],
    /// so mip 0 begins 4096 + 4096 + 16384 + 65536 = 90112 B past the base.
    #[test]
    fn base_mip_placement_handles_the_4kib_block() {
        let placement = base_mip_placement(5, 256, 256, 2, 9).expect("a 9-level chain places");
        assert_eq!(placement.byte_offset, 90_112);
        assert_eq!(placement.chain_slice_bytes, 352_256);
        assert_eq!(
            placement.byte_offset + tiled_byte_count_for_mode(5, 256, 256, 2).unwrap(),
            placement.chain_slice_bytes
        );
    }

    /// No relocation to apply is reported as `None`, never as a guessed offset:
    /// the caller then keeps the descriptor base (today's behaviour) and says so.
    #[test]
    fn base_mip_placement_refuses_instead_of_guessing() {
        assert_eq!(
            base_mip_placement(9, 512, 512, 2, 1),
            None,
            "a single-level resource has no chain"
        );
        assert_eq!(
            base_mip_placement(9, 512, 512, 2, 0),
            None,
            "MAX_MIP+1 can never be 0, but a malformed T# must not shift anything"
        );
        assert_eq!(
            base_mip_placement(1, 512, 512, 2, 10),
            None,
            "swizzle mode 1 has no ported equation"
        );
        assert_eq!(
            base_mip_placement(9, 512, 512, 5, 10),
            None,
            "32-byte elements are past the last swizzle-table row"
        );
        assert_eq!(base_mip_placement(9, 0, 512, 2, 10), None, "zero extent");
    }

    /// When the whole chain fits inside the tail block, mip 0 is a sub-rectangle
    /// of the DETILED block rather than a block grid of its own. For 64 KiB at
    /// 4 B/element the last tail slot's micro-block coordinate is (8, 0) and a
    /// micro block is 8x8 elements, so the sub-rectangle starts at (64, 0).
    #[test]
    fn base_mip_placement_finds_the_in_tail_sub_rectangle() {
        let placement = base_mip_placement(9, 64, 64, 2, 7).expect("64x64 with 7 levels is a tail");
        assert_eq!(placement.tail_element, Some((64, 0)));
        assert_eq!(
            placement.byte_offset, 0,
            "an in-tail mip 0 is inside the FIRST block"
        );
        assert_eq!(placement.chain_slice_bytes, 65_536, "one swizzle block");

        // Slot (8, 0) scales with the element size: micro blocks are 16x16 at
        // 1 B/element (block 256x256) and 4x4 at 16 B/element (block 64x64).
        assert_eq!(
            base_mip_placement(9, 64, 64, 0, 7).unwrap().tail_element,
            Some((128, 0))
        );
        assert_eq!(
            base_mip_placement(9, 32, 32, 4, 5).unwrap().tail_element,
            Some((32, 0))
        );
    }

    /// The in-tail sub-rectangle starts half a block in, so a mip 0 wider than
    /// half the block cannot fit it. That is reported as `None` — read at the
    /// descriptor base and warn — not as an out-of-block offset.
    #[test]
    fn base_mip_placement_refuses_an_in_tail_rect_that_does_not_fit() {
        assert_eq!(
            base_mip_placement(9, 128, 64, 2, 5),
            None,
            "128 elements wide + a 64-element tail offset overruns a 128-wide block"
        );
    }

    /// Functional check of the in-tail path: tile a whole 128x128 block, then lift
    /// the (64, 0) 64x64 sub-rectangle back out. Detiling the block first is what
    /// makes mip 0's rows contiguous — in the tiled bytes they are interleaved
    /// with the other tail levels.
    #[test]
    fn detile_mip_tail_base_lifts_the_sub_rectangle() {
        let (bw, bh, bpp_log2) = (128u32, 128u32, 2u32);
        let bpp = 1usize << bpp_log2;
        let linear: Vec<u8> = (0..(bw * bh) as usize * bpp)
            .map(|i| (i % 253) as u8)
            .collect();
        let tiled = tile_64kb_s(&linear, bw, bh, bpp_log2);

        let (rx, ry, rw, rh) = (64u32, 0u32, 64u32, 64u32);
        let got = detile_mip_tail_base(9, &tiled, rw, rh, bpp_log2, rx, ry)
            .expect("the sub-rectangle fits the block");
        let row = rw as usize * bpp;
        let mut want = vec![0u8; row * rh as usize];
        for y in 0..rh as usize {
            let src = ((ry as usize + y) * bw as usize + rx as usize) * bpp;
            want[y * row..(y + 1) * row].copy_from_slice(&linear[src..src + row]);
        }
        assert_eq!(got, want, "detiled block sub-rectangle");

        assert_eq!(
            detile_mip_tail_base(9, &tiled[..1024], rw, rh, bpp_log2, rx, ry),
            None,
            "a short block must be refused, not read as zeros"
        );
        assert_eq!(
            detile_mip_tail_base(9, &tiled, 128, 64, bpp_log2, rx, ry),
            None,
            "a sub-rectangle past the block edge must be refused"
        );
    }

    #[test]
    fn block_element_dimensions_names_unsupported_inputs() {
        assert_eq!(block_element_dimensions(9, 2), Some((128, 128)));
        assert_eq!(block_element_dimensions(5, 2), Some((32, 32)));
        assert_eq!(block_element_dimensions(9, 5), None, "past the last row");
        assert_eq!(block_element_dimensions(2, 2), None, "no ported equation");
    }
}
