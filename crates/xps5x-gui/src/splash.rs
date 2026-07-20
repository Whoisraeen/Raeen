//! System boot splash — what a PS5 shows between launch and the title's first
//! frame.
//!
//! The console never boots to a black screen: system software presents the
//! package's `sce_sys/pic0.png` from launch until the title calls
//! `sceSystemServiceHideSplashScreen` (or its own frames start presenting).
//! SharpEmu does the same (`PngSplashLoader.cs`), which is the entire reason it
//! "shows the ASTRO.BOT splash" while a title is still booting — no title
//! rendering is involved.
//!
//! The launcher stages the decoded image with
//! [`xps5x_gpu::AgcGpuSession::set_pending_splash`] *before* entering the
//! guest, because the process GPU session is created inside `execute_process`.
//! Every launch stages either `Some` or `None`, so a previous title's splash
//! cannot leak into the next.

use std::path::Path;
use tracing::{debug, info};

/// Decode `<game dir>/sce_sys/pic0.png` for the eboot at `eboot_path` and
/// stage it as the boot splash for the upcoming launch. A missing or
/// undecodable image stages `None` — launching proceeds identically, just
/// without a splash.
pub(crate) fn stage_boot_splash(eboot_path: &Path) {
    let splash = load_pic0(eboot_path);
    match &splash {
        Some(image) => info!(
            width = image.width,
            height = image.height,
            "boot splash: staged sce_sys/pic0.png"
        ),
        None => debug!("boot splash: no decodable sce_sys/pic0.png beside the eboot"),
    }
    xps5x_gpu::AgcGpuSession::set_pending_splash(splash);
}

fn load_pic0(eboot_path: &Path) -> Option<xps5x_gpu::RenderedImage> {
    let pic0 = eboot_path.parent()?.join("sce_sys").join("pic0.png");
    let decoded = match image::open(&pic0) {
        Ok(decoded) => decoded,
        Err(err) => {
            // Distinguish "package ships no splash" (normal for homebrew)
            // from "splash exists but did not decode" (worth a log line).
            if pic0.exists() {
                debug!(path = %pic0.display(), error = %err, "boot splash: pic0.png failed to decode");
            }
            return None;
        }
    };
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    Some(xps5x_gpu::RenderedImage {
        width,
        height,
        pixels: rgba.into_raw(),
        bytes_per_pixel: 4,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1 PNG round-trips through the loader into the RGBA byte layout the
    /// Shell presents (`from_rgba_unmultiplied`).
    #[test]
    fn pic0_decodes_to_rgba_rendered_image() {
        let dir = std::env::temp_dir().join(format!("xps5x-splash-test-{}", std::process::id()));
        let sce_sys = dir.join("sce_sys");
        std::fs::create_dir_all(&sce_sys).expect("mkdir sce_sys");
        let mut png = image::RgbaImage::new(2, 1);
        png.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        png.put_pixel(1, 0, image::Rgba([0, 0, 255, 255]));
        png.save(sce_sys.join("pic0.png")).expect("write pic0");

        let splash = load_pic0(&dir.join("eboot.bin")).expect("decodes");
        assert_eq!((splash.width, splash.height), (2, 1));
        assert_eq!(splash.bytes_per_pixel, 4);
        assert_eq!(splash.pixels, vec![255, 0, 0, 255, 0, 0, 255, 255]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No `sce_sys/pic0.png` is the homebrew norm, not an error.
    #[test]
    fn missing_pic0_is_none() {
        let dir = std::env::temp_dir().join(format!("xps5x-splash-missing-{}", std::process::id()));
        assert!(load_pic0(&dir.join("eboot.bin")).is_none());
    }
}
