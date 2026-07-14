//! On-disk theme loading (spec §6, §9, §11).
//!
//! A theme directory is untrusted content once user themes exist:
//!
//! ```text
//! themes/<name>/
//!   theme.toml     # palette + metrics tokens, optional asset refs
//!   font.ttf       # optional custom UI font (user-supplied)
//!   backgrounds/…  # optional hero/background images (user-supplied)
//! ```
//!
//! [`load_theme`] parses `theme.toml`, resolves palette/metrics
//! field-by-field against [`default_theme`], and loads the optional font
//! and background asset it references. Every asset — the manifest itself,
//! the font, the background — is bounds-checked and falls back to the
//! matching in-code default on anything missing, malformed, oversized, or
//! path-traversing. Nothing here ever panics on untrusted input, executes
//! code, or reaches outside the theme's own directory.

use super::{Theme, ThemeAssets, default_theme};
use egui::{Color32, ColorImage};
use serde::Deserialize;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Manifest file name inside a theme directory.
const MANIFEST_FILE: &str = "theme.toml";

/// Reject a `theme.toml` bigger than this before even parsing it.
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
/// Reject a font file bigger than this.
const MAX_FONT_BYTES: u64 = 8 * 1024 * 1024;
/// Reject a background image whose encoded file is bigger than this (a
/// cheap pre-decode check; [`MAX_IMAGE_DIM`]/[`MAX_IMAGE_DECODED_BYTES`]
/// additionally cap the *decoded* size via `image::Limits`).
const MAX_IMAGE_ENCODED_BYTES: u64 = 32 * 1024 * 1024;
/// Reject a background image wider or taller than this, in either the
/// declared or the decoded dimensions.
const MAX_IMAGE_DIM: u32 = 4096;
/// Reject a background image whose decoded (RGBA8) size would exceed this.
const MAX_IMAGE_DECODED_BYTES: u64 = 32 * 1024 * 1024;

/// The on-disk manifest shape. Every field is optional; a missing or
/// unparsable field simply falls back to the matching field of
/// [`default_theme`] rather than failing the whole load (spec §6: "a
/// partial or broken theme must degrade gracefully, never panic").
#[derive(Debug, Default, Deserialize)]
struct ManifestFile {
    name: Option<String>,
    #[serde(default)]
    palette: PaletteManifest,
    #[serde(default)]
    metrics: MetricsManifest,
    #[serde(default)]
    assets: AssetsManifest,
}

#[derive(Debug, Default, Deserialize)]
struct PaletteManifest {
    ground: Option<String>,
    raised: Option<String>,
    line: Option<String>,
    text: Option<String>,
    text_dim: Option<String>,
    text_faint: Option<String>,
    accent: Option<String>,
    accent_hi: Option<String>,
    focus: Option<String>,
    scrim: Option<String>,
    cc_scrim: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct MetricsManifest {
    topbar_padding_x: Option<f32>,
    topbar_padding_top: Option<f32>,
    content_padding_x: Option<f32>,
    content_padding_bottom: Option<f32>,
    tile_size: Option<f32>,
    tile_gap: Option<f32>,
    tile_focus_scale: Option<f32>,
    tile_focus_lift: Option<f32>,
    rail_padding_left: Option<f32>,
    corner_radius: Option<f32>,
    button_radius: Option<f32>,
    card_width: Option<f32>,
    card_height: Option<f32>,
    card_gap: Option<f32>,
    cc_item_size: Option<f32>,
    cc_item_gap: Option<f32>,
}

/// Asset references, resolved relative to the theme's own directory. Every
/// reference is untrusted user input — see [`resolve_asset_path`].
#[derive(Debug, Default, Deserialize)]
struct AssetsManifest {
    font: Option<String>,
    background: Option<String>,
}

/// Parse a `#RRGGBB` (opaque) or `#RRGGBBAA` literal into a [`Color32`],
/// using the same raw-byte semantics as the in-code palette (spec: the
/// on-disk and in-code defaults are required to agree exactly). Returns
/// `None` for anything else — callers fall back to the default value.
fn parse_hex_color(s: &str) -> Option<Color32> {
    let s = s.strip_prefix('#')?;
    let byte = |i: usize| u8::from_str_radix(s.get(i..i + 2)?, 16).ok();
    match s.len() {
        6 => Some(Color32::from_rgba_premultiplied(
            byte(0)?,
            byte(2)?,
            byte(4)?,
            255,
        )),
        8 => Some(Color32::from_rgba_premultiplied(
            byte(0)?,
            byte(2)?,
            byte(4)?,
            byte(6)?,
        )),
        _ => None,
    }
}

fn color_or(value: &Option<String>, fallback: Color32) -> Color32 {
    value
        .as_deref()
        .and_then(parse_hex_color)
        .unwrap_or(fallback)
}

/// A manifest float only overrides the default when it parsed *and* is
/// finite — a `theme.toml` with `tile_size = nan` or `inf` must not corrupt
/// layout math downstream.
fn f32_or(value: Option<f32>, fallback: f32) -> f32 {
    value.filter(|v| v.is_finite()).unwrap_or(fallback)
}

fn resolve_palette(m: &PaletteManifest, fallback: &super::Palette) -> super::Palette {
    super::Palette {
        ground: color_or(&m.ground, fallback.ground),
        raised: color_or(&m.raised, fallback.raised),
        line: color_or(&m.line, fallback.line),
        text: color_or(&m.text, fallback.text),
        text_dim: color_or(&m.text_dim, fallback.text_dim),
        text_faint: color_or(&m.text_faint, fallback.text_faint),
        accent: color_or(&m.accent, fallback.accent),
        accent_hi: color_or(&m.accent_hi, fallback.accent_hi),
        focus: color_or(&m.focus, fallback.focus),
        scrim: color_or(&m.scrim, fallback.scrim),
        cc_scrim: color_or(&m.cc_scrim, fallback.cc_scrim),
    }
}

fn resolve_metrics(m: &MetricsManifest, fallback: &super::Metrics) -> super::Metrics {
    super::Metrics {
        topbar_padding_x: f32_or(m.topbar_padding_x, fallback.topbar_padding_x),
        topbar_padding_top: f32_or(m.topbar_padding_top, fallback.topbar_padding_top),
        content_padding_x: f32_or(m.content_padding_x, fallback.content_padding_x),
        content_padding_bottom: f32_or(m.content_padding_bottom, fallback.content_padding_bottom),
        tile_size: f32_or(m.tile_size, fallback.tile_size),
        tile_gap: f32_or(m.tile_gap, fallback.tile_gap),
        tile_focus_scale: f32_or(m.tile_focus_scale, fallback.tile_focus_scale),
        tile_focus_lift: f32_or(m.tile_focus_lift, fallback.tile_focus_lift),
        rail_padding_left: f32_or(m.rail_padding_left, fallback.rail_padding_left),
        corner_radius: f32_or(m.corner_radius, fallback.corner_radius),
        button_radius: f32_or(m.button_radius, fallback.button_radius),
        card_size: egui::vec2(
            f32_or(m.card_width, fallback.card_size.x),
            f32_or(m.card_height, fallback.card_size.y),
        ),
        card_gap: f32_or(m.card_gap, fallback.card_gap),
        cc_item_size: f32_or(m.cc_item_size, fallback.cc_item_size),
        cc_item_gap: f32_or(m.cc_item_gap, fallback.cc_item_gap),
    }
}

/// Read `path` in full, but only if it exists, is a regular file, and is no
/// larger than `max_bytes`. Returns `None` — never panics, never partially
/// reads — for a missing file, an oversized file, or any I/O error; every
/// caller treats `None` as "fall back to default".
fn read_capped(path: &Path, max_bytes: u64) -> Option<Vec<u8>> {
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    if metadata.len() > max_bytes {
        tracing::warn!(path = %path.display(), size = metadata.len(), max = max_bytes, "theme asset exceeds size cap — ignoring");
        return None;
    }
    std::fs::read(path).ok()
}

/// Resolve a manifest-declared asset reference to a path inside
/// `theme_dir`. Rejects absolute paths and any `..` component *before*
/// touching the filesystem, then re-verifies containment after
/// canonicalization so a symlink inside the theme directory can't be used
/// to escape it either (spec §6, §11: theme directories are untrusted).
fn resolve_asset_path(theme_dir: &Path, reference: &str) -> Option<PathBuf> {
    let rel = Path::new(reference);
    // `has_root()` (not just `is_absolute()`) also catches a Unix-style
    // `/etc/passwd` reference on Windows, where it "has root" but isn't
    // technically `is_absolute()` (no drive prefix) — `PathBuf::join` would
    // otherwise still resolve it against the current drive's root.
    if rel.is_absolute() || rel.has_root() {
        tracing::warn!(reference, "theme asset reference is absolute — rejecting");
        return None;
    }
    if rel.components().any(|c| matches!(c, Component::ParentDir)) {
        tracing::warn!(reference, "theme asset reference contains '..' — rejecting");
        return None;
    }
    let candidate = theme_dir.join(rel);
    let base = theme_dir.canonicalize().ok()?;
    let resolved = candidate.canonicalize().ok()?;
    if resolved.starts_with(&base) {
        Some(resolved)
    } else {
        tracing::warn!(
            reference,
            "theme asset reference escapes the theme directory — rejecting"
        );
        None
    }
}

fn load_font(theme_dir: &Path, reference: &str) -> Option<Vec<u8>> {
    let path = resolve_asset_path(theme_dir, reference)?;
    read_capped(&path, MAX_FONT_BYTES)
}

fn load_background(theme_dir: &Path, reference: &str) -> Option<ColorImage> {
    let path = resolve_asset_path(theme_dir, reference)?;
    load_image_file_capped(&path)
}

/// Read + decode an image file under the same size caps as a theme
/// background. Shared with the Shell's per-game cover loading (`shell/
/// mod.rs`) — covers are user-supplied untrusted content exactly like theme
/// backgrounds, so they get the identical bounds-checked path. Returns
/// `None` — never panics — for anything missing, oversized, or malformed.
pub(crate) fn load_image_file_capped(path: &Path) -> Option<ColorImage> {
    let bytes = read_capped(path, MAX_IMAGE_ENCODED_BYTES)?;
    decode_image_capped(&bytes)
}

/// `true` if `w`x`h` is small enough to decode/hold as an in-memory RGBA8
/// image under [`MAX_IMAGE_DIM`]/[`MAX_IMAGE_DECODED_BYTES`]. Split out as
/// its own pure function so the size-cap logic is unit-testable without
/// needing to decode real image bytes.
fn dims_within_limits(w: u32, h: u32) -> bool {
    if w == 0 || h == 0 || w > MAX_IMAGE_DIM || h > MAX_IMAGE_DIM {
        return false;
    }
    let decoded_bytes = (w as u64) * (h as u64) * 4;
    decoded_bytes <= MAX_IMAGE_DECODED_BYTES
}

/// Decode `bytes` as an image through `image`'s safe decoders, bounded by
/// [`image::Limits`] (belt) and a redundant post-decode dimension check
/// (suspenders — spec §6/§11 asks for bounds-checked decoding, not just
/// "trust the crate"). Returns `None` — never panics — for anything
/// malformed, oversized, or of an unsupported format.
fn decode_image_capped(bytes: &[u8]) -> Option<ColorImage> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_DECODED_BYTES);

    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    reader.limits(limits);
    let decoded = reader.decode().ok()?;

    let (w, h) = (decoded.width(), decoded.height());
    if !dims_within_limits(w, h) {
        tracing::warn!(
            width = w,
            height = h,
            "theme background image exceeds dimension/size cap — ignoring"
        );
        return None;
    }

    let rgba = decoded.into_rgba8();
    Some(ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        rgba.as_raw(),
    ))
}

/// Load the theme named `name` from `themes_root/<name>/theme.toml`,
/// falling back to [`default_theme`] wholesale if the manifest is missing,
/// oversized, or not valid TOML, and field-by-field (palette entries,
/// metrics, font, background) for anything narrower that's missing or
/// invalid within an otherwise-valid manifest.
pub fn load_theme(themes_root: &Path, name: &str) -> Theme {
    let default = default_theme();
    let theme_dir = themes_root.join(name);
    let manifest_path = theme_dir.join(MANIFEST_FILE);

    let manifest = read_capped(&manifest_path, MAX_MANIFEST_BYTES)
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|text| toml::from_str::<ManifestFile>(&text).ok());

    let Some(manifest) = manifest else {
        return default;
    };

    let palette = resolve_palette(&manifest.palette, &default.palette);
    let metrics = resolve_metrics(&manifest.metrics, &default.metrics);
    let font = manifest
        .assets
        .font
        .as_deref()
        .and_then(|reference| load_font(&theme_dir, reference));
    let background = manifest
        .assets
        .background
        .as_deref()
        .and_then(|reference| load_background(&theme_dir, reference));

    Theme {
        name: manifest.name.unwrap_or(default.name),
        palette,
        metrics,
        assets: ThemeAssets {
            font: font.map(Arc::from),
            background: background.map(Arc::new),
        },
    }
}

/// Enumerate installed theme directories under `themes_root` — always
/// including `"default"` even if the directory doesn't (yet) exist on disk,
/// since the Shell always has an in-code default to fall back to.
pub fn list_themes(themes_root: &Path) -> Vec<String> {
    let mut names = vec!["default".to_string()];
    if let Ok(entries) = std::fs::read_dir(themes_root) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let Some(dir_name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !names.contains(&dir_name) {
                names.push(dir_name);
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &Path, rel: &str, contents: &[u8]) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    /// A fresh scratch directory under the OS temp dir — avoids depending
    /// on a temp-dir crate (spec §11: no new deps beyond `image`) while
    /// keeping test fixtures (including intentionally-invalid ones, like
    /// path-traversal targets) well outside the repository tree.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("xps5x-gui-theme-loader-tests")
            .join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_theme_dir_falls_back_to_default() {
        let root = scratch_dir("missing");
        let theme = load_theme(&root, "does-not-exist");
        assert_eq!(theme.name, default_theme().name);
        assert_eq!(theme.palette.ground, default_theme().palette.ground);
        assert!(theme.assets.font.is_none());
        assert!(theme.assets.background.is_none());
    }

    #[test]
    fn valid_manifest_overrides_declared_tokens() {
        let root = scratch_dir("valid");
        write_file(
            &root,
            "custom/theme.toml",
            br##"
                name = "Custom"

                [palette]
                ground = "#101010"
                accent = "#ff00ff"

                [metrics]
                tile_size = 200.0
                card_width = 300.0
                card_height = 50.0
            "##,
        );

        let theme = load_theme(&root, "custom");
        assert_eq!(theme.name, "Custom");
        assert_eq!(
            theme.palette.ground,
            Color32::from_rgba_premultiplied(0x10, 0x10, 0x10, 255)
        );
        assert_eq!(
            theme.palette.accent,
            Color32::from_rgba_premultiplied(0xff, 0x00, 0xff, 255)
        );
        assert_eq!(theme.metrics.tile_size, 200.0);
        assert_eq!(theme.metrics.card_size, egui::vec2(300.0, 50.0));

        // Everything not declared in the manifest falls back to the default.
        let default = default_theme();
        assert_eq!(theme.palette.raised, default.palette.raised);
        assert_eq!(theme.metrics.tile_gap, default.metrics.tile_gap);
    }

    #[test]
    fn malformed_hex_and_missing_fields_fall_back_to_default() {
        let root = scratch_dir("malformed");
        write_file(
            &root,
            "broken/theme.toml",
            br##"
                [palette]
                ground = "not-a-color"
                accent = "#zzzzzz"

                [metrics]
                tile_size = "nope"
            "##,
        );

        // `tile_size = "nope"` is a type mismatch for an `f32` field, which
        // makes the whole manifest fail to parse — the loader must still
        // degrade to the full default rather than panicking.
        let theme = load_theme(&root, "broken");
        let default = default_theme();
        assert_eq!(theme.palette.ground, default.palette.ground);
        assert_eq!(theme.metrics.tile_size, default.metrics.tile_size);
    }

    #[test]
    fn malformed_hex_within_an_otherwise_valid_manifest_falls_back_per_field() {
        let root = scratch_dir("malformed-partial");
        write_file(
            &root,
            "partial/theme.toml",
            br##"
                [palette]
                ground = "not-a-color"
                accent = "#00ff00"
            "##,
        );

        let theme = load_theme(&root, "partial");
        let default = default_theme();
        assert_eq!(
            theme.palette.ground, default.palette.ground,
            "invalid hex falls back to default"
        );
        assert_eq!(
            theme.palette.accent,
            Color32::from_rgba_premultiplied(0x00, 0xff, 0x00, 255),
            "valid hex is honored"
        );
    }

    #[test]
    fn oversized_manifest_falls_back_to_default() {
        let root = scratch_dir("oversized-manifest");
        let huge = vec![b' '; (MAX_MANIFEST_BYTES + 1) as usize];
        write_file(&root, "huge/theme.toml", &huge);

        let theme = load_theme(&root, "huge");
        assert_eq!(theme.palette.ground, default_theme().palette.ground);
    }

    #[test]
    fn parent_dir_asset_reference_is_rejected() {
        let root = scratch_dir("traversal-parent");
        std::fs::create_dir_all(root.join("victim")).unwrap();
        write_file(&root, "secret.ttf", b"not a real font");
        write_file(
            &root,
            "victim/theme.toml",
            br##"[assets]
font = "../secret.ttf""##,
        );

        let theme = load_theme(&root, "victim");
        assert!(
            theme.assets.font.is_none(),
            "'..' asset references must be rejected, not read"
        );
    }

    #[test]
    fn absolute_path_asset_reference_is_rejected() {
        let root = scratch_dir("traversal-absolute");
        write_file(
            &root,
            "victim/theme.toml",
            br##"[assets]
background = "/etc/passwd""##,
        );

        let theme = load_theme(&root, "victim");
        assert!(
            theme.assets.background.is_none(),
            "absolute asset references must be rejected, not read"
        );
    }

    #[test]
    fn resolve_asset_path_accepts_a_reference_that_stays_inside_the_theme_dir() {
        let root = scratch_dir("resolve-ok");
        write_file(&root, "font.ttf", b"font bytes");
        let resolved = resolve_asset_path(&root, "font.ttf");
        assert!(resolved.is_some());
        assert!(resolved.unwrap().starts_with(root.canonicalize().unwrap()));
    }

    #[test]
    fn oversized_font_falls_back_to_no_font() {
        let root = scratch_dir("oversized-font");
        let huge_font = vec![0u8; (MAX_FONT_BYTES + 1) as usize];
        write_file(&root, "big/font.ttf", &huge_font);
        write_file(
            &root,
            "big/theme.toml",
            br##"[assets]
font = "font.ttf""##,
        );

        let theme = load_theme(&root, "big");
        assert!(theme.assets.font.is_none());
    }

    #[test]
    fn missing_referenced_asset_falls_back_to_none_without_panicking() {
        let root = scratch_dir("missing-asset");
        write_file(
            &root,
            "ghost/theme.toml",
            br##"[assets]
font = "does-not-exist.ttf"
background = "backgrounds/does-not-exist.png""##,
        );

        let theme = load_theme(&root, "ghost");
        assert!(theme.assets.font.is_none());
        assert!(theme.assets.background.is_none());
    }

    #[test]
    fn dims_within_limits_boundary_values() {
        assert!(dims_within_limits(1, 1));
        // Comfortably under both the dimension cap and the decoded-bytes cap.
        assert!(dims_within_limits(2000, 2000));
        assert!(
            !dims_within_limits(0, 10),
            "zero-sized dimensions are rejected"
        );
        assert!(
            !dims_within_limits(10, 0),
            "zero-sized dimensions are rejected"
        );
        assert!(
            !dims_within_limits(MAX_IMAGE_DIM + 1, 10),
            "width over the dimension cap is rejected"
        );
        assert!(
            !dims_within_limits(10, MAX_IMAGE_DIM + 1),
            "height over the dimension cap is rejected"
        );
        // Exactly at the dimension cap on both axes still exceeds the
        // decoded-bytes cap (4096 * 4096 * 4 > 32 MiB) — both limits apply.
        assert!(!dims_within_limits(MAX_IMAGE_DIM, MAX_IMAGE_DIM));
    }

    /// Generates a tiny valid PNG in-memory (via the `image` crate itself)
    /// to exercise the real decode path end-to-end, not just the pure
    /// dimension-check helper.
    #[test]
    fn decode_image_capped_decodes_a_small_valid_png() {
        let img = image::RgbaImage::from_pixel(4, 3, image::Rgba([10, 20, 30, 255]));
        let mut bytes: Vec<u8> = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut bytes);
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
        }

        let decoded = decode_image_capped(&bytes).expect("valid PNG must decode");
        assert_eq!(decoded.size, [4, 3]);
        assert_eq!(
            decoded.pixels[0],
            Color32::from_rgba_unmultiplied(10, 20, 30, 255)
        );
    }

    #[test]
    fn decode_image_capped_rejects_garbage_bytes() {
        assert!(decode_image_capped(b"this is not an image").is_none());
    }

    #[test]
    fn valid_background_image_loads_through_the_full_theme() {
        let root = scratch_dir("with-background");
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]));
        let mut bytes: Vec<u8> = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut bytes);
            image::DynamicImage::ImageRgba8(img)
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
        }
        write_file(&root, "bg/backgrounds/hero.png", &bytes);
        write_file(
            &root,
            "bg/theme.toml",
            br##"[assets]
background = "backgrounds/hero.png""##,
        );

        let theme = load_theme(&root, "bg");
        let background = theme.assets.background.expect("background must load");
        assert_eq!(background.size, [2, 2]);
    }

    #[test]
    fn list_themes_always_includes_default_plus_discovered_dirs() {
        let root = scratch_dir("list");
        std::fs::create_dir_all(root.join("midnight")).unwrap();
        std::fs::create_dir_all(root.join("retro")).unwrap();
        // A stray file (not a directory) must not be listed as a theme.
        let mut f = std::fs::File::create(root.join("readme.txt")).unwrap();
        f.write_all(b"not a theme").unwrap();

        let names = list_themes(&root);
        assert!(names.contains(&"default".to_string()));
        assert!(names.contains(&"midnight".to_string()));
        assert!(names.contains(&"retro".to_string()));
        assert!(!names.contains(&"readme.txt".to_string()));
    }

    #[test]
    fn list_themes_on_a_nonexistent_root_still_returns_default() {
        let names = list_themes(Path::new("this/does/not/exist/at/all"));
        assert_eq!(names, vec!["default".to_string()]);
    }

    /// The repository's own `themes/default/theme.toml` (spec §11: original
    /// values only, no binary assets) must parse to *exactly* the in-code
    /// [`default_theme`] — the two are required to agree so the on-disk
    /// default is a faithful, inspectable record of the built-in one.
    #[test]
    fn repo_default_theme_toml_matches_the_in_code_default() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let themes_root = workspace_root.join("themes");
        let loaded = load_theme(&themes_root, "default");
        let expected = default_theme();

        assert_eq!(loaded.name, expected.name);
        assert_eq!(loaded.palette, expected.palette);
        assert_eq!(loaded.metrics, expected.metrics);
        assert!(
            loaded.assets.font.is_none(),
            "the shipped default theme carries no binary assets"
        );
        assert!(
            loaded.assets.background.is_none(),
            "the shipped default theme carries no binary assets"
        );
    }
}
