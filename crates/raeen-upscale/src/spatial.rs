//! CPU spatial resamplers + sharpen — the backends that genuinely run on the
//! current present ABI (which hands plugins tightly-packed CPU pixels).
//!
//! High-quality filtering is done for the 4-byte display formats
//! (RGBA8/BGRA8/packed B10G11R11). For other `bytes_per_pixel` (e.g. 8-byte
//! R16G16B16A16 HDR, whose bytes are half-floats that must not be blended as
//! `u8`) we fall back to a safe nearest-neighbour replicate.

/// Which spatial kernel a [`resample`] call uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// 2×2 linear — cheap, soft.
    Bilinear,
    /// 4×4 Catmull-Rom bicubic — sharper edges, the default "quality" path.
    Bicubic,
}

/// Resample `src` (`sw`×`sh`, `bpp` bytes/pixel, row-major, no padding) to
/// `dw`×`dh`. Returns the destination buffer (`dw*dh*bpp` bytes).
#[must_use]
pub fn resample(
    src: &[u8],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    bpp: u32,
    kernel: Kernel,
) -> Vec<u8> {
    if bpp == 4 && sw > 0 && sh > 0 {
        match kernel {
            Kernel::Bilinear => bilinear_rgba8(src, sw, sh, dw, dh),
            Kernel::Bicubic => bicubic_rgba8(src, sw, sh, dw, dh),
        }
    } else {
        nearest(src, sw, sh, dw, dh, bpp)
    }
}

/// Safe format-agnostic fallback: nearest-neighbour, whole-texel copies.
#[must_use]
pub fn nearest(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32, bpp: u32) -> Vec<u8> {
    let bpp = bpp as usize;
    let mut out = vec![0u8; dw as usize * dh as usize * bpp];
    if sw == 0 || sh == 0 {
        return out;
    }
    for y in 0..dh {
        let sy = ((y as u64 * sh as u64) / dh as u64).min(sh as u64 - 1) as u32;
        for x in 0..dw {
            let sx = ((x as u64 * sw as u64) / dw as u64).min(sw as u64 - 1) as u32;
            let s = ((sy * sw + sx) as usize) * bpp;
            let d = ((y * dw + x) as usize) * bpp;
            if s + bpp <= src.len() {
                out[d..d + bpp].copy_from_slice(&src[s..s + bpp]);
            }
        }
    }
    out
}

#[inline]
fn px(src: &[u8], sw: u32, sh: u32, x: i64, y: i64) -> [f32; 4] {
    let x = x.clamp(0, sw as i64 - 1) as u32;
    let y = y.clamp(0, sh as i64 - 1) as u32;
    let o = ((y * sw + x) as usize) * 4;
    match src.get(o..o + 4) {
        Some(p) => [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32],
        None => [0.0; 4],
    }
}

fn bilinear_rgba8(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; dw as usize * dh as usize * 4];
    let rx = sw as f32 / dw as f32;
    let ry = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = (y as f32 + 0.5) * ry - 0.5;
        let y0 = fy.floor() as i64;
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = (x as f32 + 0.5) * rx - 0.5;
            let x0 = fx.floor() as i64;
            let tx = fx - x0 as f32;
            let c00 = px(src, sw, sh, x0, y0);
            let c10 = px(src, sw, sh, x0 + 1, y0);
            let c01 = px(src, sw, sh, x0, y0 + 1);
            let c11 = px(src, sw, sh, x0 + 1, y0 + 1);
            let d = ((y * dw + x) as usize) * 4;
            for c in 0..4 {
                let top = c00[c] + (c10[c] - c00[c]) * tx;
                let bot = c01[c] + (c11[c] - c01[c]) * tx;
                out[d + c] = (top + (bot - top) * ty).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Catmull-Rom cubic weight (a = -0.5).
#[inline]
fn catmull(t: f32) -> f32 {
    const A: f32 = -0.5;
    let t = t.abs();
    if t <= 1.0 {
        (A + 2.0) * t * t * t - (A + 3.0) * t * t + 1.0
    } else if t < 2.0 {
        A * t * t * t - 5.0 * A * t * t + 8.0 * A * t - 4.0 * A
    } else {
        0.0
    }
}

fn bicubic_rgba8(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; dw as usize * dh as usize * 4];
    let rx = sw as f32 / dw as f32;
    let ry = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = (y as f32 + 0.5) * ry - 0.5;
        let iy = fy.floor() as i64;
        let wy = [
            catmull(fy - (iy - 1) as f32),
            catmull(fy - iy as f32),
            catmull(fy - (iy + 1) as f32),
            catmull(fy - (iy + 2) as f32),
        ];
        for x in 0..dw {
            let fx = (x as f32 + 0.5) * rx - 0.5;
            let ix = fx.floor() as i64;
            let wx = [
                catmull(fx - (ix - 1) as f32),
                catmull(fx - ix as f32),
                catmull(fx - (ix + 1) as f32),
                catmull(fx - (ix + 2) as f32),
            ];
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for (j, wyj) in wy.iter().enumerate() {
                for (i, wxi) in wx.iter().enumerate() {
                    let w = wxi * wyj;
                    let p = px(src, sw, sh, ix - 1 + i as i64, iy - 1 + j as i64);
                    for c in 0..4 {
                        acc[c] += p[c] * w;
                    }
                    wsum += w;
                }
            }
            let d = ((y * dw + x) as usize) * 4;
            let inv = if wsum.abs() > 1e-6 { 1.0 / wsum } else { 1.0 };
            for c in 0..4 {
                out[d + c] = (acc[c] * inv).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// In-place unsharp sharpen for a 4-byte-per-pixel image (RGB channels only;
/// alpha left untouched). `amount` ~0.0..=1.0. No-op for other formats.
#[must_use]
pub fn sharpen(src: &[u8], w: u32, h: u32, bpp: u32, amount: f32) -> Vec<u8> {
    if bpp != 4 || w == 0 || h == 0 || amount <= 0.0 {
        return src.to_vec();
    }
    let mut out = src.to_vec();
    let k = amount.clamp(0.0, 2.0);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let c = px(src, w, h, x, y);
            let n = px(src, w, h, x, y - 1);
            let s = px(src, w, h, x, y + 1);
            let e = px(src, w, h, x + 1, y);
            let ww = px(src, w, h, x - 1, y);
            let d = ((y as u32 * w + x as u32) as usize) * 4;
            for ch in 0..3 {
                let sharp = c[ch] + k * (4.0 * c[ch] - n[ch] - s[ch] - e[ch] - ww[ch]);
                out[d + ch] = sharp.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_doubles_dimensions_and_preserves_corners() {
        // 2x2 distinct pixels → 4x4. Bilinear/bicubic keep the corner colors.
        let src = vec![
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 0, 255, // (1,1) yellow
        ];
        for k in [Kernel::Bilinear, Kernel::Bicubic] {
            let out = resample(&src, 2, 2, 4, 4, 4, k);
            assert_eq!(out.len(), 4 * 4 * 4);
            // Top-left stays red-dominant.
            assert!(out[0] > 200 && out[1] < 80 && out[2] < 80, "{k:?} TL");
        }
    }

    #[test]
    fn non_rgba8_falls_back_to_nearest_safely() {
        // 8-byte (HDR) pixels: must not blend bytes; nearest replicate.
        // Source 2x1, destination 4x1, 8 bytes per texel.
        let src = vec![1u8; 2 * 8];
        let out = resample(&src, 2, 1, 4, 1, 8, Kernel::Bicubic);
        assert_eq!(out.len(), 4 * 8);
        assert!(out.iter().all(|&b| b == 1));
    }

    #[test]
    fn sharpen_is_identity_at_zero_amount_and_changes_pixels_otherwise() {
        let src = vec![
            10, 10, 10, 255, 200, 200, 200, 255, 10, 10, 10, 255, 200, 200, 200, 255,
        ];
        assert_eq!(sharpen(&src, 2, 2, 4, 0.0), src);
        let sharpened = sharpen(&src, 2, 2, 4, 0.6);
        assert_ne!(sharpened, src, "a non-zero amount must change pixels");
        assert_eq!(sharpened.len(), src.len());
    }
}
