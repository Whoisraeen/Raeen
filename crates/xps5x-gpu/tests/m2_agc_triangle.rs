//! M2 acceptance: AGC PM4 DRAW → Vulkan triangle (kyty-graphics SPIR-V).
//!
//! This is the CLAUDE.md M2 gate: a Gen5 DCB with `DRAW_INDEX_AUTO` drives an
//! offscreen rasterize whose pixels are verified by readback, and a PPM is
//! written for screenshot inspection.
//!
//! Machines without Vulkan 1.3 skip (unless `XPS5X_REQUIRE_VULKAN=1`).
//!
//! ```text
//! XPS5X_REQUIRE_VULKAN=1 cargo test -p xps5x-gpu --test m2_agc_triangle -- --nocapture
//! ```

// This gate deliberately exercises the deprecated no-register fixture path
// (`build_m2_draw_dcb`/`execute_dcb`): it is the M2 regression check that the
// simplest DCB still rasterizes, distinct from the register-driven CP path.
#![allow(deprecated)]

use xps5x_gpu::agc_exec::{AgcGpuSession, M2_DRAW_HEIGHT, M2_DRAW_WIDTH, build_m2_draw_dcb};
use xps5x_gpu::vulkan::{CLEAR_COLOR, TRIANGLE_COLOR, unorm8, validation_error_count};

const TOLERANCE: u8 = 2;

fn assert_pixel_eq(actual: [u8; 4], expected: [u8; 4], label: &str) {
    let close = actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE);
    assert!(
        close,
        "{label}: expected RGBA {expected:?} (+/-{TOLERANCE}), read back {actual:?}"
    );
}

fn require_or_skip(err: &impl std::fmt::Display) -> bool {
    if std::env::var_os("XPS5X_REQUIRE_VULKAN").is_some() {
        panic!("XPS5X_REQUIRE_VULKAN is set but M2 draw failed: {err}");
    }
    eprintln!("m2_agc_triangle: SKIP — {err}");
    true
}

fn write_ppm(path: &str, width: u32, height: u32, pixels: &[u8]) {
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for chunk in pixels.chunks_exact(4) {
        ppm.extend_from_slice(&chunk[..3]);
    }
    std::fs::write(path, &ppm).expect("write PPM");
    eprintln!("m2_agc_triangle: wrote {path}");
}

/// The M2 gate: fixture AGC DCB → decode → Vulkan draw → pixel assertions.
#[test]
fn agc_pm4_draw_produces_visible_triangle() {
    let words = build_m2_draw_dcb();
    let session = AgcGpuSession::global();
    let draws_before = session.draw_count();

    let image = match session.execute_dcb(&words) {
        Ok(Some(image)) => image,
        Ok(None) => panic!("fixture DCB must contain a draw packet"),
        Err(e) => {
            if require_or_skip(&e) {
                return;
            }
            unreachable!();
        }
    };

    assert_eq!(image.width, M2_DRAW_WIDTH);
    assert_eq!(image.height, M2_DRAW_HEIGHT);
    assert_eq!(
        image.pixels.len(),
        (M2_DRAW_WIDTH * M2_DRAW_HEIGHT * 4) as usize
    );
    assert!(
        session.draw_count() > draws_before,
        "session must count the PM4-triggered draw"
    );

    let center = image
        .pixel(M2_DRAW_WIDTH / 2, M2_DRAW_HEIGHT / 2)
        .expect("center in bounds");
    assert_pixel_eq(
        center,
        unorm8(TRIANGLE_COLOR),
        "center should be the triangle",
    );

    for (x, y) in [
        (0, 0),
        (M2_DRAW_WIDTH - 1, 0),
        (0, M2_DRAW_HEIGHT - 1),
        (M2_DRAW_WIDTH - 1, M2_DRAW_HEIGHT - 1),
    ] {
        let corner = image.pixel(x, y).expect("corner in bounds");
        assert_pixel_eq(corner, unorm8(CLEAR_COLOR), &format!("corner ({x}, {y})"));
    }

    assert_ne!(unorm8(TRIANGLE_COLOR), unorm8(CLEAR_COLOR));
    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the M2 draw"
    );

    let path = std::env::var("XPS5X_TRIANGLE_PPM").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("xps5x_m2_triangle.ppm")
            .to_string_lossy()
            .into_owned()
    });
    write_ppm(&path, image.width, image.height, &image.pixels);
}

#[test]
fn agc_pm4_triangle_covers_a_plausible_pixel_fraction() {
    let words = build_m2_draw_dcb();
    let image = match AgcGpuSession::global().execute_dcb(&words) {
        Ok(Some(image)) => image,
        Ok(None) => panic!("fixture DCB must contain a draw packet"),
        Err(e) => {
            if require_or_skip(&e) {
                return;
            }
            unreachable!();
        }
    };

    let expected = unorm8(TRIANGLE_COLOR);
    let mut hits = 0u32;
    for y in 0..image.height {
        for x in 0..image.width {
            let p = image.pixel(x, y).expect("in bounds");
            if p.iter()
                .zip(expected.iter())
                .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE)
            {
                hits += 1;
            }
        }
    }
    let fraction = f64::from(hits) / f64::from(image.width * image.height);
    assert!(
        (0.15..0.35).contains(&fraction),
        "triangle should cover ~24.5% of the image, covered {:.1}% ({hits} px)",
        fraction * 100.0
    );
}
