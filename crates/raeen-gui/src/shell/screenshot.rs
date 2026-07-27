//! Screenshot capture — dump the currently published guest frame to disk.
//!
//! F12 (or the pad's Create button, the PS5's own screenshot button) writes
//! the frame the Shell is already presenting — the published RGBA image the
//! [`super::present::GameFrameView`] draws — to `screenshots/
//! <title-id>_<timestamp>.png`. This is deliberately the GUEST frame, not a
//! capture of the Shell window: what the emulator rendered, letterbox-free,
//! at the title's own resolution. There is no OS-level window capture here
//! (out of scope); with no published frame the caller shows an informational
//! toast instead.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Repo-root-relative directory screenshots are written into (like `themes/`
/// and `sounds/`). Created on first use.
pub(crate) const SCREENSHOTS_ROOT: &str = "screenshots";

/// `<sanitized-id>_<UTC timestamp>.png`. Timestamp is
/// `YYYYMMDD-HHMMSS-mmm` (UTC — no timezone database in the dependency tree,
/// and an unambiguous name beats a local-looking one that shifts with DST).
/// Millisecond precision keeps rapid consecutive captures from colliding.
pub(crate) fn file_name(title_id: &str, since_epoch: Duration) -> String {
    let (y, mo, d, h, mi, s) = utc_datetime(since_epoch.as_secs());
    format!(
        "{}_{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}-{:03}.png",
        super::ledger::sanitize_id(title_id),
        since_epoch.subsec_millis(),
    )
}

/// Civil UTC date/time from unix seconds (Howard Hinnant's `civil_from_days`,
/// era-based). Valid across the whole unix era; pure so the file-name format
/// is testable against known epochs.
fn utc_datetime(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    );
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d, h, mi, s)
}

/// Write `image` to `dir/<title-id>_<timestamp>.png`, creating `dir` first.
/// Refuses (with a human-readable reason for the failure toast) anything that
/// is not a complete 8-bit RGBA frame — the only format the present path
/// publishes today; HDR (8 bytes/pixel) would need a tonemap pass this
/// deliberately does not fake.
pub(crate) fn save(
    image: &raeen_gpu::RenderedImage,
    dir: &Path,
    title_id: &str,
) -> Result<PathBuf, String> {
    if image.bytes_per_pixel != 4 {
        return Err(format!(
            "unsupported frame format ({} bytes/pixel — only 8-bit RGBA is captured)",
            image.bytes_per_pixel
        ));
    }
    let expected = image.width as usize * image.height as usize * 4;
    if image.width == 0 || image.height == 0 || image.pixels.len() != expected {
        return Err("frame is empty or truncated".to_string());
    }
    std::fs::create_dir_all(dir).map_err(|err| format!("could not create {dir:?}: {err}"))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let path = dir.join(file_name(title_id, now));
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|err| format!("PNG write failed: {err}"))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_formats_known_epochs_in_utc() {
        // 1_000_000_000 = 2001-09-09T01:46:40Z.
        assert_eq!(
            file_name("PPSA01234", Duration::new(1_000_000_000, 0)),
            "PPSA01234_20010909-014640-000.png"
        );
        // 1_234_567_890 = 2009-02-13T23:31:30Z, with millisecond precision.
        assert_eq!(
            file_name("game", Duration::new(1_234_567_890, 250_000_000)),
            "game_20090213-233130-250.png"
        );
        // The epoch itself.
        assert_eq!(file_name("x", Duration::ZERO), "x_19700101-000000-000.png");
    }

    #[test]
    fn file_name_sanitizes_hostile_title_ids() {
        let name = file_name("../evil: id?", Duration::ZERO);
        assert_eq!(name, ".._evil__id__19700101-000000-000.png");
        assert!(!name.contains('/') && !name.contains('\\') && !name.contains(':'));
    }

    #[test]
    fn save_writes_a_decodable_png_and_creates_the_directory() {
        let dir = std::env::temp_dir().join(format!("raeen-shot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let image = raeen_gpu::RenderedImage {
            width: 2,
            height: 2,
            // Distinct corner pixels so a decode round-trip proves ordering.
            pixels: vec![
                255, 0, 0, 255, // (0,0) red
                0, 255, 0, 255, // (1,0) green
                0, 0, 255, 255, // (0,1) blue
                255, 255, 255, 255, // (1,1) white
            ],
            bytes_per_pixel: 4,
        };
        let path = save(&image, &dir, "TestGame").expect("save");
        assert!(path.exists());
        assert!(
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("TestGame_") && n.ends_with(".png"))
        );
        let back = image::open(&path).expect("decode").into_rgba8();
        assert_eq!(back.dimensions(), (2, 2));
        assert_eq!(back.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(back.get_pixel(1, 1).0, [255, 255, 255, 255]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_refuses_non_rgba8_and_truncated_frames() {
        let dir = std::env::temp_dir().join("raeen-shot-refuse");
        let hdr = raeen_gpu::RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![0; 8],
            bytes_per_pixel: 8,
        };
        assert!(save(&hdr, &dir, "t").is_err());
        let truncated = raeen_gpu::RenderedImage {
            width: 2,
            height: 2,
            pixels: vec![0; 4], // needs 16
            bytes_per_pixel: 4,
        };
        assert!(save(&truncated, &dir, "t").is_err());
        let empty = raeen_gpu::RenderedImage {
            width: 0,
            height: 0,
            pixels: Vec::new(),
            bytes_per_pixel: 4,
        };
        assert!(save(&empty, &dir, "t").is_err());
    }
}
