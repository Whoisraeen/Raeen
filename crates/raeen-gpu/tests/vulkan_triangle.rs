//! M2 acceptance test: a real Vulkan draw, verified by pixel readback.
//!
//! This renders one triangle offscreen on the host GPU and asserts on the
//! pixels that actually came back from the device. No window, no swapchain —
//! it runs headless.
//!
//! ## Skipping vs. failing
//!
//! Machines without a Vulkan 1.3 device (most CI runners) **skip**: the test
//! prints why and returns green. That would be a great way to accidentally
//! claim a passing GPU test that never ran, so setting `RAEEN_REQUIRE_VULKAN=1`
//! turns a skip into a hard failure. Use it to prove the test really executed:
//!
//! ```text
//! RAEEN_REQUIRE_VULKAN=1 cargo test -p raeen-gpu --test vulkan_triangle -- --nocapture
//! ```

use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::{
    CLEAR_COLOR, TRIANGLE_COLOR, VulkanBackend, unorm8, validation_error_count,
};

/// R8G8B8A8_UNORM quantization is allowed a little slack by the Vulkan spec
/// (conversion must land within 0.6 ULP), so exact equality is too strict for
/// the fractional clear color. This is far tighter than the gap between the
/// clear color and the triangle color, so it cannot mask a wrong pixel.
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

/// Bring up the backend, or return `None` when this machine has no usable
/// Vulkan device (unless `RAEEN_REQUIRE_VULKAN` demands one).
fn backend_or_skip() -> Option<VulkanBackend> {
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => {
            let name = backend
                .device()
                .map(|d| d.device_name().to_owned())
                .unwrap_or_default();
            let validation = backend.device().is_some_and(|d| d.validation_enabled());
            eprintln!("vulkan_triangle: running on {name} (validation={validation})");
            Some(backend)
        }
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("vulkan_triangle: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// The M2 gate: the GPU rasterized a triangle and we can see it in the pixels.
///
/// Center is the triangle's color; all four corners are the clear color. Both
/// halves matter — the center alone would also pass if the draw wrongly filled
/// the whole target, and the corners alone would pass if nothing drew at all.
#[test]
fn triangle_is_visible_in_readback_pixels() {
    let Some(backend) = backend_or_skip() else {
        return;
    };

    const W: u32 = 64;
    const H: u32 = 64;

    let image = backend
        .render_test_triangle(W, H)
        .expect("offscreen triangle render must succeed on a working device");

    assert_eq!(image.width, W);
    assert_eq!(image.height, H);
    assert_eq!(
        image.pixels.len(),
        (W * H * 4) as usize,
        "readback must be tightly packed RGBA8"
    );

    let center = image
        .pixel(W / 2, H / 2)
        .expect("center pixel is in bounds");
    assert_pixel_eq(
        center,
        unorm8(TRIANGLE_COLOR),
        "center should be the triangle",
    );

    for (x, y) in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)] {
        let corner = image.pixel(x, y).expect("corner pixel is in bounds");
        assert_pixel_eq(corner, unorm8(CLEAR_COLOR), &format!("corner ({x}, {y})"));
    }

    // The clear color and triangle color must actually differ, or the two
    // assertions above would both pass on a uniformly-filled image.
    assert_ne!(
        unorm8(TRIANGLE_COLOR),
        unorm8(CLEAR_COLOR),
        "test is meaningless if the triangle matches the clear color"
    );

    // Correct pixels are not enough: the draw must also be valid Vulkan. The
    // validation layer logs via `tracing`, which this binary has no subscriber
    // for, so assert on the counter instead of trusting the log.
    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the draw"
    );
}

/// The triangle must occupy a plausible share of the image — not one stray
/// pixel, not the whole target. Guards against a degenerate or clipped draw
/// that still happens to color the center.
#[test]
fn triangle_covers_a_plausible_pixel_fraction() {
    let Some(backend) = backend_or_skip() else {
        return;
    };

    const W: u32 = 64;
    const H: u32 = 64;
    let image = backend
        .render_test_triangle(W, H)
        .expect("offscreen triangle render must succeed on a working device");

    let expected = unorm8(TRIANGLE_COLOR);
    let mut hits = 0u32;
    for y in 0..H {
        for x in 0..W {
            let p = image.pixel(x, y).expect("pixel in bounds");
            if p.iter()
                .zip(expected.iter())
                .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE)
            {
                hits += 1;
            }
        }
    }

    // The NDC triangle spans 1.4 x 1.4 of a 2.0 x 2.0 clip area: area
    // 0.5 * 1.4 * 1.4 = 0.98 of 4.0, i.e. ~24.5% of the image. Bounds are wide
    // enough to absorb rasterization rules but tight enough to catch a
    // fully-covered or barely-covered target.
    let fraction = f64::from(hits) / f64::from(W * H);
    assert!(
        (0.15..0.35).contains(&fraction),
        "triangle should cover ~24.5% of the image, covered {:.1}% ({hits} px)",
        fraction * 100.0
    );
}

/// Diagnostic: dump the rendered triangle so a human can look at it.
///
/// The M2 gate asks for a *screenshotable* triangle, and pixel assertions alone
/// cannot tell a triangle from any other shape with similar coverage. Writes a
/// binary PPM (P6) — no image-crate dependency needed — and an ASCII preview.
/// Ignored by default because it is an eyeball tool, not a gate:
///
/// ```text
/// cargo test -p raeen-gpu --test vulkan_triangle -- --ignored --nocapture
/// ```
///
/// Set `RAEEN_TRIANGLE_PPM` to choose the output path.
#[test]
#[ignore = "diagnostic: writes an image for manual inspection"]
fn dump_triangle_image() {
    let Some(backend) = backend_or_skip() else {
        return;
    };

    const W: u32 = 48;
    const H: u32 = 48;
    let image = backend
        .render_test_triangle(W, H)
        .expect("offscreen triangle render must succeed on a working device");

    let path = std::env::var("RAEEN_TRIANGLE_PPM").unwrap_or_else(|_| "triangle.ppm".to_owned());
    let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
    for y in 0..H {
        for x in 0..W {
            let p = image.pixel(x, y).expect("pixel in bounds");
            ppm.extend_from_slice(&p[..3]); // PPM is RGB, drop alpha
        }
    }
    std::fs::write(&path, &ppm).expect("write PPM");
    eprintln!("wrote {path}");

    let expected = unorm8(TRIANGLE_COLOR);
    for y in 0..H {
        let row: String = (0..W)
            .map(|x| {
                let p = image.pixel(x, y).expect("pixel in bounds");
                let hit = p
                    .iter()
                    .zip(expected.iter())
                    .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE);
                if hit { '#' } else { '.' }
            })
            .collect();
        eprintln!("{row}");
    }
}

/// Rendering before `init()` must be a clean error, not a panic or a crash in
/// the driver.
#[test]
fn render_before_init_is_an_error() {
    let backend = VulkanBackend::new(false);
    assert!(!backend.is_initialized());
    assert!(
        backend.render_test_triangle(8, 8).is_err(),
        "rendering without a device must fail cleanly"
    );
}
