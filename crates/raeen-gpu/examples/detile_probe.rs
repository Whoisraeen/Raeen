//! Offline diagnostic for replaying a captured guest texture through Raeen's
//! supported GFX10 detilers. This never runs in the emulator hot path.

use std::error::Error;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: detile_probe <rgba.bin> <width> <height>")?,
    );
    let width: u32 = args
        .next()
        .ok_or("missing width")?
        .to_string_lossy()
        .parse()?;
    let height: u32 = args
        .next()
        .ok_or("missing height")?
        .to_string_lossy()
        .parse()?;
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let source = std::fs::read(&input)?;
    for mode in [5, 9, 24, 27] {
        let Some(linear) = raeen_gpu::texture::tiling::detile_64kb(mode, &source, width, height, 2)
        else {
            continue;
        };
        write_rgb_ppm(&input, mode, width, height, &linear)?;
    }
    let oberon = detile_oberon_render_32(&source, width, height);
    write_rgb_ppm(&input, 127, width, height, &oberon)?;
    Ok(())
}

/// KytyPS5's independently derived Prospero `RenderTarget64KB` 32-bpp
/// equation (`guest_gpu/tile.cpp::Gen5RenderTargetOffsetInBlock`). Mode 127 is
/// only an output label for this diagnostic; it is not a guest swizzle mode.
fn detile_oberon_render_32(source: &[u8], width: u32, height: u32) -> Vec<u8> {
    const BLOCK_WIDTH: u32 = 128;
    const BLOCK_HEIGHT: u32 = 128;
    const BLOCK_BYTES: usize = 65_536;
    let blocks_per_row = width.div_ceil(BLOCK_WIDTH) as usize;
    let mut linear = vec![0; width as usize * height as usize * 4];
    for y in 0..height {
        for x in 0..width {
            let block = (y / BLOCK_HEIGHT) as usize * blocks_per_row + (x / BLOCK_WIDTH) as usize;
            let mut offset = 0u32;
            offset ^= (y << 4) & 0x0070;
            offset ^= (y << 5) & 0x0f00;
            offset ^= (y << 9) & 0x1000;
            offset ^= (y << 8) & 0x4000;
            offset ^= (x << 2) & 0x000c;
            offset ^= (x << 5) & 0x0380;
            offset ^= (x << 4) & 0x0400;
            offset ^= (x << 6) & 0x0800;
            offset ^= (x << 9) & 0xa000;
            let src = block * BLOCK_BYTES + offset as usize;
            let dst = (y as usize * width as usize + x as usize) * 4;
            if src + 4 <= source.len() {
                linear[dst..dst + 4].copy_from_slice(&source[src..src + 4]);
            }
        }
    }
    linear
}

fn write_rgb_ppm(
    input: &Path,
    mode: u8,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), Box<dyn Error>> {
    let required = width as usize * height as usize * 4;
    if rgba.len() < required {
        return Err(format!("mode {mode}: {} bytes, need {required}", rgba.len()).into());
    }
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in rgba[..required].chunks_exact(4) {
        ppm.extend_from_slice(&pixel[..3]);
    }
    let stem = input
        .file_stem()
        .ok_or("input has no filename")?
        .to_string_lossy();
    let output = input.with_file_name(format!("{stem}_detile-m{mode}.ppm"));
    std::fs::write(&output, ppm)?;
    println!("{}", output.display());
    Ok(())
}
