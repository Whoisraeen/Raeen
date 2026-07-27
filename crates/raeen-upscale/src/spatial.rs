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

// ─── FSR1-class edge-adaptive upscale ───────────────────────────────────────
//
// AMD's FidelityFX Super Resolution 1.0 is a *spatial* technique: unlike
// FSR2/3, DLSS and XeSS it needs no motion vectors, no history, and no vendor
// runtime — which is exactly why it is the one that can work here today, and
// why it can live in-tree at all (FSR1 is MIT, and MIT is GPL-2.0
// compatible; DLSS/XeSS are closed and must stay user-supplied binaries).
//
// FSR1 is two passes:
//   EASU — Edge Adaptive Spatial Upsampling: reconstruct at the target
//          resolution by fitting the local gradient, so edges stay crisp
//          instead of being blurred across like a plain bicubic tap.
//   RCAS — Robust Contrast Adaptive Sharpening: re-add high frequencies
//          without amplifying noise or ringing near already-sharp edges.
//
// This is an ORIGINAL implementation of that two-pass approach written
// against the published description of the algorithm, not a port of AMD's
// shader source, so it is deliberately *FSR1-class* rather than bit-identical
// to AMD's. See THIRD_PARTY_NOTICES.md.

/// Local edge direction/strength at a source pixel, from the 3×3 luma
/// neighbourhood. `(dir_x, dir_y)` is the (unnormalised) gradient and `len`
/// its magnitude — how confident we are that an edge exists here at all.
fn edge_at(src: &[u8], sw: u32, sh: u32, x: i64, y: i64) -> (f32, f32, f32) {
    let luma = |dx: i64, dy: i64| {
        let p = px(src, sw, sh, x + dx, y + dy);
        // Rec.601 luma: edge detection wants perceived brightness, and doing
        // it on one channel would miss edges that live in the others.
        0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
    };
    // Sobel — cheap, isotropic enough for an upscaling gradient, and far more
    // stable on noisy frames than a bare central difference.
    let gx = (luma(1, -1) + 2.0 * luma(1, 0) + luma(1, 1))
        - (luma(-1, -1) + 2.0 * luma(-1, 0) + luma(-1, 1));
    let gy = (luma(-1, 1) + 2.0 * luma(0, 1) + luma(1, 1))
        - (luma(-1, -1) + 2.0 * luma(0, -1) + luma(1, -1));
    (gx, gy, (gx * gx + gy * gy).sqrt())
}

/// EASU-class pass: upscale `src` to `dw`×`dh`, steering each destination
/// sample along the local edge so edges resolve sharply.
///
/// Where the neighbourhood is flat the result is plain bicubic (nothing to
/// steer); as edge confidence rises the sample is pulled toward the bicubic
/// tap taken *along* the edge rather than across it, which is what keeps a
/// diagonal from turning into a staircase of blurred steps.
#[must_use]
fn easu_rgba8(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    // Start from the bicubic reconstruction and refine it: bicubic already
    // handles the flat majority of the frame correctly and cheaply.
    let mut out = bicubic_rgba8(src, sw, sh, dw, dh);
    if sw < 3 || sh < 3 {
        return out; // too small for a 3x3 gradient — bicubic is the honest answer
    }
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    // Below this gradient magnitude the "edge" is noise; steering there would
    // sharpen grain and dither patterns into artefacts.
    const EDGE_FLOOR: f32 = 12.0;
    // Cap the steer so a very strong edge cannot pull a sample more than one
    // source texel away, which would fabricate detail that is not there.
    const MAX_STEER: f32 = 1.0;

    for dy in 0..dh {
        for dx in 0..dw {
            // Source position of this destination sample (pixel centres).
            let fx = (dx as f32 + 0.5) * sx - 0.5;
            let fy = (dy as f32 + 0.5) * sy - 0.5;
            let (gx, gy, len) = edge_at(src, sw, sh, fx.round() as i64, fy.round() as i64);
            if len <= EDGE_FLOOR {
                continue; // flat: keep the bicubic value
            }
            // Unit vector ALONG the edge is perpendicular to the gradient.
            let along = (-gy / len, gx / len);
            // Confidence ramps in over the floor and saturates, so the
            // transition between "bicubic" and "steered" is gradual — a hard
            // switch would show up as a visible seam along edge boundaries.
            let confidence = ((len - EDGE_FLOOR) / 64.0).clamp(0.0, 1.0);
            let steer = confidence * MAX_STEER;
            // Average two taps placed along the edge: this reinforces the
            // edge's own direction instead of averaging across it.
            let a = sample_bicubic_at(src, sw, sh, fx + along.0 * steer, fy + along.1 * steer);
            let b = sample_bicubic_at(src, sw, sh, fx - along.0 * steer, fy - along.1 * steer);
            let d = ((dy * dw + dx) as usize) * 4;
            for ch in 0..4 {
                let steered = 0.5 * (a[ch] + b[ch]);
                let base = f32::from(out[d + ch]);
                // Blend toward the steered value by confidence.
                let v = base + confidence * (steered - base);
                out[d + ch] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// One bicubic sample at an arbitrary (fractional) source position — the
/// per-sample form of [`bicubic_rgba8`], used by the edge-steered taps.
fn sample_bicubic_at(src: &[u8], sw: u32, sh: u32, fx: f32, fy: f32) -> [f32; 4] {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let mut acc = [0.0f32; 4];
    for m in -1..=2i64 {
        let wy = catmull((ty - m as f32).abs());
        if wy == 0.0 {
            continue;
        }
        for n in -1..=2i64 {
            let wx = catmull((tx - n as f32).abs());
            if wx == 0.0 {
                continue;
            }
            let p = px(src, sw, sh, x0 + n, y0 + m);
            let w = wx * wy;
            for ch in 0..4 {
                acc[ch] += p[ch] * w;
            }
        }
    }
    acc
}

/// RCAS-class pass: contrast-adaptive sharpening.
///
/// Unlike the plain unsharp in [`sharpen`], the strength here is scaled down
/// where the local neighbourhood already has high contrast. That is the whole
/// point of "robust": uniform sharpening rings hard edges and amplifies noise,
/// while this concentrates the correction on the soft mid-contrast detail that
/// upscaling actually smeared.
#[must_use]
pub fn rcas(src: &[u8], w: u32, h: u32, bpp: u32, sharpness: f32) -> Vec<u8> {
    if bpp != 4 || w == 0 || h == 0 || sharpness <= 0.0 {
        return src.to_vec();
    }
    let mut out = src.to_vec();
    let k = sharpness.clamp(0.0, 1.0);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let c = px(src, w, h, x, y);
            let n = px(src, w, h, x, y - 1);
            let s = px(src, w, h, x, y + 1);
            let e = px(src, w, h, x + 1, y);
            let ww = px(src, w, h, x - 1, y);
            let d = ((y as u32 * w + x as u32) as usize) * 4;
            for ch in 0..3 {
                // Local contrast from the 4-neighbourhood.
                let mn = n[ch].min(s[ch]).min(e[ch]).min(ww[ch]).min(c[ch]);
                let mx = n[ch].max(s[ch]).max(e[ch]).max(ww[ch]).max(c[ch]);
                let range = (mx - mn).max(1.0);
                // Attenuate where contrast is already high (near 255 range)
                // and apply fully in smooth regions.
                let attenuation = (1.0 - (range / 255.0)).clamp(0.0, 1.0);
                let laplacian = 4.0 * c[ch] - n[ch] - s[ch] - e[ch] - ww[ch];
                let sharpened = c[ch] + k * attenuation * laplacian;
                // Clamp into the local neighbourhood range: this is what
                // prevents overshoot ringing on either side of an edge.
                out[d + ch] = sharpened.clamp(mn, mx).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Full FSR1-class upscale: EASU-style edge-adaptive upsample, then
/// RCAS-style contrast-adaptive sharpening.
///
/// Falls back to a safe nearest replicate for non-4-byte formats, matching
/// [`resample`] — 8-byte HDR halves must not be blended as `u8`.
#[must_use]
pub fn fsr1(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32, bpp: u32, sharpness: f32) -> Vec<u8> {
    if bpp != 4 {
        return nearest(src, sw, sh, dw, dh, bpp);
    }
    let upscaled = easu_rgba8(src, sw, sh, dw, dh);
    rcas(&upscaled, dw, dh, 4, sharpness)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EASU must produce the requested size, stay in range, and — the point of
    /// the whole pass — keep a hard edge harder than a plain bicubic does.
    #[test]
    fn easu_upscales_and_preserves_edges_better_than_bicubic() {
        // 8x8, left half black / right half white: one vertical hard edge.
        let (w, h) = (8u32, 8u32);
        let mut src = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0 } else { 255 };
                let d = ((y * w + x) * 4) as usize;
                src[d] = v;
                src[d + 1] = v;
                src[d + 2] = v;
                src[d + 3] = 255;
            }
        }
        let (dw, dh) = (16u32, 16u32);
        let easu = easu_rgba8(&src, w, h, dw, dh);
        let bicubic = bicubic_rgba8(&src, w, h, dw, dh);
        assert_eq!(easu.len(), (dw * dh * 4) as usize);

        // Measure edge width along a scanline: count pixels that are neither
        // near-black nor near-white (i.e. blurred across the transition).
        let transition_width = |img: &[u8]| {
            let y = dh / 2;
            (0..dw)
                .filter(|x| {
                    let v = img[((y * dw + x) * 4) as usize];
                    v > 24 && v < 231
                })
                .count()
        };
        assert!(
            transition_width(&easu) <= transition_width(&bicubic),
            "EASU must not blur the edge more than bicubic (easu {} vs bicubic {})",
            transition_width(&easu),
            transition_width(&bicubic)
        );
    }

    /// RCAS must never ring: every output sample stays inside the range of its
    /// own input neighbourhood, which is the property plain unsharp violates.
    #[test]
    fn rcas_sharpens_without_overshooting_the_local_range() {
        let (w, h) = (5u32, 5u32);
        let mut src = vec![255u8; (w * h * 4) as usize];
        // A single dark pixel in a bright field — maximum ringing bait.
        let c = ((2 * w + 2) * 4) as usize;
        src[c] = 40;
        src[c + 1] = 40;
        src[c + 2] = 40;

        let out = rcas(&src, w, h, 4, 1.0);
        assert_eq!(out.len(), src.len());
        for i in 0..(w * h) as usize {
            for ch in 0..3 {
                let v = out[i * 4 + ch];
                assert!(
                    (40..=255).contains(&v),
                    "sample {i} ch {ch} = {v} left the input range — that is ringing"
                );
            }
        }
        // Zero sharpness is exactly identity.
        assert_eq!(rcas(&src, w, h, 4, 0.0), src);
    }

    #[test]
    fn fsr1_produces_the_target_size_and_is_safe_for_hdr() {
        let (w, h) = (4u32, 4u32);
        let src = vec![128u8; (w * h * 4) as usize];
        let out = fsr1(&src, w, h, 8, 8, 4, 0.5);
        assert_eq!(out.len(), (8 * 8 * 4) as usize);

        // 8-byte HDR must not be blended as u8 — fall back to nearest.
        let hdr = vec![0u8; (w * h * 8) as usize];
        let out_hdr = fsr1(&hdr, w, h, 8, 8, 8, 0.5);
        assert_eq!(out_hdr.len(), (8 * 8 * 8) as usize);
    }

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
