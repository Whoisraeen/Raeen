//! Game-folder scanning + per-game metadata parsing (spec §4, §9, §10).
//!
//! [`item_from_folder`] is the pure classifier: given a folder name, its
//! path, whether an `eboot.bin` was found inside it, and the raw text of an
//! optional `xps5x-title.toml` alongside it, decide whether the folder is a
//! valid game install and build a [`LibraryItem`] for it. It touches no
//! filesystem state itself, so it's unit-testable without fixtures on disk.
//! [`scan_dir`] is the thin IO wrapper that walks a real directory, reads
//! each folder's optional metadata file, and calls the classifier.
//!
//! [`parse_title_meta`] is the pure metadata parser: given the text of an
//! `xps5x-title.toml`, return the parsed title/metadata/gradient, or `None`
//! for malformed TOML or a file missing the required `title` field. A
//! malformed or absent metadata file is never fatal — the folder is still a
//! valid game, just without metadata (title falls back to the folder name,
//! art falls back to an id-derived gradient).

use super::{
    ActivityCard, GameMeta, ItemKind, LaunchTarget, LibraryItem, art_from_id, art_from_stops,
};
use egui::Color32;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The on-disk shape of `xps5x-title.toml`. All fields except `title` are
/// optional and default sensibly when absent.
///
/// TOML authoring note: like any root-level key, `gradient` must be written
/// *before* any `[[activity]]` table — TOML attaches a bare `key = value`
/// line to the most recently opened table, so a `gradient` line placed
/// after an `[[activity]]` entry silently becomes that entry's field
/// instead of the document root's (and is then ignored, since
/// `ActivityToml` has no such field).
#[derive(Debug, Deserialize)]
struct TitleToml {
    title: String,
    #[serde(default)]
    genre: String,
    #[serde(default)]
    players: String,
    #[serde(default)]
    rating: u8,
    #[serde(default = "default_ready")]
    ready: String,
    /// Total play time as display text ("2h 1m") — Home's "Time played"
    /// stat. Absent → shown as an em-dash.
    #[serde(default)]
    time_played: String,
    /// Most recently earned trophy name — Home's "Last trophy" stat.
    /// Absent → shown as an em-dash.
    #[serde(default)]
    last_trophy: String,
    #[serde(default)]
    activity: Vec<ActivityToml>,
    /// Two hex colors (`"#rrggbb"`, `"0xrrggbb"`, or `"rrggbb"`), bright
    /// stop first. Absent → art is derived deterministically from the game
    /// id instead (spec §4).
    #[serde(default)]
    gradient: Option<[String; 2]>,
}

fn default_ready() -> String {
    "Ready to play".to_string()
}

#[derive(Debug, Deserialize)]
struct ActivityToml {
    kind: String,
    #[serde(default)]
    main: String,
    #[serde(default)]
    sub: String,
    #[serde(default)]
    progress: Option<u8>,
}

/// The result of successfully parsing an `xps5x-title.toml` string.
pub struct ParsedTitleMeta {
    pub title: String,
    pub meta: GameMeta,
    /// Explicit (bright, dark) gradient stops, if the file specified one.
    pub gradient: Option<(Color32, Color32)>,
}

/// Parse an `xps5x-title.toml` string. Returns `None` for malformed TOML or
/// a file missing the required `title` field — callers fall back to a
/// folder-derived title with no metadata; this never panics on bad input.
pub fn parse_title_meta(contents: &str) -> Option<ParsedTitleMeta> {
    let raw: TitleToml = toml::from_str(contents).ok()?;
    if raw.title.trim().is_empty() {
        return None;
    }

    let gradient = raw
        .gradient
        .as_ref()
        .and_then(|[hi, lo]| Some((parse_hex_color(hi)?, parse_hex_color(lo)?)));

    let activity = raw
        .activity
        .into_iter()
        .map(|a| ActivityCard {
            top: a.kind,
            main: a.main,
            sub: a.sub,
            progress: a.progress.map(|p| p.min(100)),
        })
        .collect();

    Some(ParsedTitleMeta {
        title: raw.title,
        meta: GameMeta {
            genre: raw.genre,
            players: raw.players,
            rating: raw.rating.min(5),
            kicker: raw.ready,
            time_played: raw.time_played,
            last_trophy: raw.last_trophy,
            activity,
        },
        gradient,
    })
}

/// Parse a `#rrggbb` / `0xrrggbb` / `rrggbb` hex color literal. Returns
/// `None` for anything else rather than panicking — an invalid gradient
/// stop just falls back to id-derived art, it doesn't invalidate the rest
/// of an otherwise-valid metadata file.
fn parse_hex_color(s: &str) -> Option<Color32> {
    let s = s.trim();
    let s = s
        .strip_prefix('#')
        .or_else(|| s.strip_prefix("0x"))
        .unwrap_or(s);
    if s.len() != 6 || !s.is_ascii() {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some(Color32::from_rgb(r, g, b))
}

/// Classify a single game folder. Returns `None` if it isn't a valid game
/// install (no `eboot.bin` found, or an empty name). `title_toml` is the
/// raw text of `<folder>/xps5x-title.toml`, if one exists and was
/// readable — parsing failures fall back to a folder-derived title and no
/// metadata rather than skipping the game.
pub fn item_from_folder(
    name: &str,
    eboot: Option<PathBuf>,
    title_toml: Option<&str>,
) -> Option<LibraryItem> {
    let eboot = eboot?;
    if name.trim().is_empty() {
        return None;
    }

    let id = name.to_string();
    let parsed = title_toml.and_then(parse_title_meta);

    let (title, meta, art) = match parsed {
        Some(p) => {
            let art = match p.gradient {
                Some((hi, lo)) => art_from_stops(hi, lo),
                None => art_from_id(&id),
            };
            (p.title, Some(p.meta), art)
        }
        None => (derive_title(name), None, art_from_id(&id)),
    };

    Some(LibraryItem {
        id,
        title,
        kind: ItemKind::Game,
        art,
        meta,
        cover_path: None,
        title_id: None,
        version: None,
        launch: LaunchTarget::Game { path: eboot },
    })
}

/// Enrich a scanned item with the title's own `sce_sys/param.json` (next to
/// the eboot): the real title name, title id, and content version every
/// installed PS5 title ships. The user's `xps5x-title.toml` (explicit
/// intent) still outranks it for the display title; the folder-derived
/// fallback does not. Absent or unparsable metadata changes nothing.
fn enrich_from_param_json(item: &mut LibraryItem, eboot: &Path, had_title_toml: bool) {
    let Some(dir) = eboot.parent() else { return };
    let Ok(metadata) = xps5x_loader::pkg::scan_game_directory(dir) else {
        return;
    };
    // A title with no ASCII at all (e.g. a Japanese-only default locale)
    // would render as tofu boxes — the Shell font ships no CJK glyphs. Keep
    // the readable folder-derived title in that case.
    let renderable = metadata.title.chars().any(|c| c.is_ascii_alphanumeric());
    if !metadata.title.is_empty() && metadata.title != "Unknown" && !had_title_toml && renderable {
        item.title = metadata.title;
    }
    if !metadata.title_id.is_empty() && metadata.title_id != "UNKNOWN00000" {
        item.title_id = Some(metadata.title_id);
    }
    if !metadata.app_version.is_empty() {
        item.version = Some(metadata.app_version);
    }
}

/// Locate a game folder's `eboot.bin`.
///
/// Two layouts are supported, because both occur in practice:
/// * `<game>/eboot.bin` — the flat layout this project's fixtures use.
/// * `<game>/<TITLEID>-app/eboot.bin` — how a **real PS5 title installs**
///   (e.g. `Minecraft/PPSA17221-app/eboot.bin`). Without this, every real game
///   folder scans as "no eboot" and never appears in the library.
///
/// The nested search is one level deep and prefers a conventional `*-app`
/// directory, falling back to any single subdirectory that holds an
/// `eboot.bin`, so it stays predictable rather than walking the whole tree.
fn find_eboot(dir: &Path) -> Option<PathBuf> {
    let flat = dir.join("eboot.bin");
    if flat.is_file() {
        return Some(flat);
    }

    let mut fallback = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let sub = entry.path();
        if !sub.is_dir() {
            continue;
        }
        let candidate = sub.join("eboot.bin");
        if !candidate.is_file() {
            continue;
        }
        let is_app_dir = sub
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("-app"));
        if is_app_dir {
            return Some(candidate);
        }
        fallback.get_or_insert(candidate);
    }
    fallback
}

/// Conventional cover-image file names looked for inside a game folder, in
/// priority order. User-supplied, like theme backgrounds — the repository
/// itself never ships cover images (spec §11).
const COVER_FILE_NAMES: [&str; 3] = ["cover.png", "cover.jpg", "cover.jpeg"];

/// The tile art for a game. A user-supplied `cover.*` in the game's own folder
/// wins (so anyone can override), otherwise the title's **own** app icon —
/// `sce_sys/icon0.png`, which every real PS5 title ships next to its eboot.
/// `None` leaves the tile on its generated gradient + monogram (e.g. a
/// decrypted dump with no `sce_sys`).
fn find_cover(folder: &Path, eboot: Option<&Path>) -> Option<std::path::PathBuf> {
    COVER_FILE_NAMES
        .iter()
        .map(|name| folder.join(name))
        .find(|p| p.is_file())
        .or_else(|| sce_sys_asset(eboot, "icon0.png"))
}

/// The full-bleed key-art background for a game: the title's own
/// `sce_sys/pic1.png` (dedicated background art) if present, otherwise
/// `pic0.png` (the boot splash, which is full-frame key art for most titles).
/// Both live in `sce_sys/` next to the eboot. `None` leaves the Home hero on
/// its generated gradient.
pub fn title_background(eboot: &Path) -> Option<std::path::PathBuf> {
    sce_sys_asset(Some(eboot), "pic1.png").or_else(|| sce_sys_asset(Some(eboot), "pic0.png"))
}

/// A file inside the title's `sce_sys/` directory, which sits next to the
/// eboot. Real titles install as `<game>/<TITLEID>-app/eboot.bin` with art
/// under `<game>/<TITLEID>-app/sce_sys/`, so the eboot's parent is the correct
/// anchor for both the flat and nested layouts [`find_eboot`] accepts.
fn sce_sys_asset(eboot: Option<&Path>, name: &str) -> Option<std::path::PathBuf> {
    let path = eboot?.parent()?.join("sce_sys").join(name);
    path.is_file().then_some(path)
}

/// Turn a folder name like `nova-requiem` or `sable_horizon` into a display
/// title like "Nova Requiem" / "Sable Horizon".
fn derive_title(folder_name: &str) -> String {
    folder_name
        .split(['-', '_', ' '])
        .filter(|word| !word.is_empty())
        .map(title_case_word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_case_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Walk `root` one level deep, classifying each subdirectory as a potential
/// game install and reading its optional `xps5x-title.toml`. Non-directories,
/// unreadable roots, and folders without an `eboot.bin` are skipped, never
/// fatal (spec §9).
pub fn scan_dir(root: &Path) -> Vec<LibraryItem> {
    let mut items = Vec::new();

    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return items,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let eboot = find_eboot(&path);
        let title_toml = std::fs::read_to_string(path.join("xps5x-title.toml")).ok();
        if let Some(mut item) = item_from_folder(name, eboot.clone(), title_toml.as_deref()) {
            item.cover_path = find_cover(&path, eboot.as_deref());
            if let Some(eboot) = eboot.as_deref() {
                enrich_from_param_json(&mut item, eboot, title_toml.is_some());
            }
            items.push(item);
        }
    }

    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const VALID_TOML: &str = r##"
        title = "Nova Requiem"
        genre = "Action RPG"
        players = "Single-player"
        rating = 5
        ready = "Ready to play"
        time_played = "2h 1m"
        last_trophy = "Hollow Walker"
        gradient = ["#ff4d6d", "#7a1338"]

        [[activity]]
        kind = "Continue"
        main = "Chapter 4 — The Hollow"
        sub = "2h ago"

        [[activity]]
        kind = "Trophies"
        main = "58%"
        sub = "24 / 41"
        progress = 58
    "##;

    #[test]
    fn parses_a_valid_title_toml() {
        let parsed = parse_title_meta(VALID_TOML).expect("should parse");
        assert_eq!(parsed.title, "Nova Requiem");
        assert_eq!(parsed.meta.genre, "Action RPG");
        assert_eq!(parsed.meta.players, "Single-player");
        assert_eq!(parsed.meta.rating, 5);
        assert_eq!(parsed.meta.kicker, "Ready to play");
        assert_eq!(parsed.meta.time_played, "2h 1m");
        assert_eq!(parsed.meta.last_trophy, "Hollow Walker");
        assert_eq!(parsed.meta.progress_percent(), Some(58));
        assert_eq!(parsed.meta.activity.len(), 2);
        assert_eq!(parsed.meta.activity[0].top, "Continue");
        assert_eq!(parsed.meta.activity[0].main, "Chapter 4 — The Hollow");
        assert_eq!(parsed.meta.activity[1].progress, Some(58));
        let (hi, lo) = parsed.gradient.expect("gradient should parse");
        assert_eq!(hi, Color32::from_rgb(0xff, 0x4d, 0x6d));
        assert_eq!(lo, Color32::from_rgb(0x7a, 0x13, 0x38));
    }

    #[test]
    fn malformed_toml_is_skipped_not_panicking() {
        assert!(parse_title_meta("this is not valid = = toml [[[").is_none());
    }

    #[test]
    fn missing_title_field_is_treated_as_malformed() {
        assert!(parse_title_meta("genre = \"Action\"").is_none());
    }

    #[test]
    fn empty_title_field_is_treated_as_malformed() {
        assert!(parse_title_meta("title = \"   \"").is_none());
    }

    #[test]
    fn missing_optional_fields_default_sensibly() {
        let parsed = parse_title_meta("title = \"Kingfall\"").expect("should parse");
        assert_eq!(parsed.title, "Kingfall");
        assert_eq!(parsed.meta.genre, "");
        assert_eq!(parsed.meta.players, "");
        assert_eq!(parsed.meta.rating, 0);
        assert_eq!(parsed.meta.kicker, "Ready to play");
        assert_eq!(parsed.meta.time_played, "");
        assert_eq!(parsed.meta.last_trophy, "");
        assert_eq!(parsed.meta.progress_percent(), None);
        assert!(parsed.meta.activity.is_empty());
        assert!(parsed.gradient.is_none());
    }

    #[test]
    fn rating_above_five_is_clamped() {
        let parsed = parse_title_meta("title = \"Overrated\"\nrating = 9").expect("should parse");
        assert_eq!(parsed.meta.rating, 5);
    }

    #[test]
    fn invalid_gradient_hex_falls_back_to_no_gradient_without_failing_whole_file() {
        let parsed =
            parse_title_meta("title = \"Bad Gradient\"\ngradient = [\"not-a-color\", \"#000000\"]")
                .expect("rest of the file is still valid");
        assert!(parsed.gradient.is_none());
        assert_eq!(parsed.title, "Bad Gradient");
    }

    #[test]
    fn hex_color_accepts_hash_and_0x_and_bare_forms() {
        assert_eq!(
            parse_hex_color("#ff0000"),
            Some(Color32::from_rgb(0xff, 0, 0))
        );
        assert_eq!(
            parse_hex_color("0x00ff00"),
            Some(Color32::from_rgb(0, 0xff, 0))
        );
        assert_eq!(
            parse_hex_color("0000ff"),
            Some(Color32::from_rgb(0, 0, 0xff))
        );
        assert_eq!(parse_hex_color("nope"), None);
        assert_eq!(parse_hex_color("#ff00"), None);
    }

    #[test]
    fn folder_with_eboot_and_valid_metadata_uses_it() {
        let path = PathBuf::from("Games/nova-requiem");
        let item = item_from_folder(
            "nova-requiem",
            Some(path.join("eboot.bin")),
            Some(VALID_TOML),
        )
        .expect("should classify");
        assert_eq!(item.title, "Nova Requiem");
        assert_eq!(item.kind, ItemKind::Game);
        let meta = item.meta.expect("metadata should be attached");
        assert_eq!(meta.genre, "Action RPG");
        match item.launch {
            LaunchTarget::Game { path } => {
                assert_eq!(path, PathBuf::from("Games/nova-requiem/eboot.bin"))
            }
            _ => panic!("expected a Game launch target"),
        }
    }

    #[test]
    fn folder_with_eboot_and_no_metadata_still_becomes_a_game() {
        let path = PathBuf::from("Games/nova-requiem");
        let item = item_from_folder("nova-requiem", Some(path.join("eboot.bin")), None)
            .expect("should classify");
        assert_eq!(item.title, "Nova Requiem");
        assert_eq!(item.kind, ItemKind::Game);
        assert!(item.meta.is_none());
    }

    #[test]
    fn folder_with_eboot_and_malformed_metadata_still_becomes_a_game() {
        let path = PathBuf::from("Games/nova-requiem");
        let item = item_from_folder(
            "nova-requiem",
            Some(path.join("eboot.bin")),
            Some("not valid toml [[["),
        )
        .expect("should classify");
        assert_eq!(item.title, "Nova Requiem"); // falls back to folder-derived title
        assert!(item.meta.is_none());
    }

    #[test]
    fn folder_without_eboot_is_skipped() {
        assert!(item_from_folder("not-a-game", None, None).is_none());
    }

    #[test]
    fn empty_name_is_skipped_even_with_eboot() {
        let path = PathBuf::from("Games/");
        assert!(item_from_folder("", Some(path.join("eboot.bin")), None).is_none());
        assert!(item_from_folder("   ", Some(path.join("eboot.bin")), None).is_none());
    }

    #[test]
    fn title_derivation_handles_separators_and_casing() {
        assert_eq!(derive_title("sable_horizon"), "Sable Horizon");
        assert_eq!(derive_title("astral-drift"), "Astral Drift");
        assert_eq!(derive_title("kingfall"), "Kingfall");
        assert_eq!(derive_title("neon verge"), "Neon Verge");
    }

    #[test]
    fn scan_dir_on_missing_root_returns_empty_not_panic() {
        let items = scan_dir(&PathBuf::from("this/path/does/not/exist/anywhere"));
        assert!(items.is_empty());
    }

    /// A fresh scratch directory under the OS temp dir (same pattern as the
    /// theme loader's tests) — cover detection needs real files on disk.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("xps5x-gui-scan-tests").join(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A **real PS5 title** installs as `<game>/<TITLEID>-app/eboot.bin`, not
    /// `<game>/eboot.bin`. The scanner used to look only at the flat path, so
    /// every real game folder classified as "no eboot" and never appeared in
    /// the library at all. Both layouts must work, and the launch target must
    /// point at the eboot that was actually found.
    #[test]
    fn scan_dir_finds_a_real_ps5_titles_nested_eboot() {
        let root = std::env::temp_dir().join(format!("xps5x-scan-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let app = root.join("Minecraft").join("PPSA17221-app");
        std::fs::create_dir_all(&app).expect("create nested app dir");
        std::fs::write(app.join("eboot.bin"), b"\x54\x14\xf5\xee").expect("write eboot");

        let items = scan_dir(&root);
        assert_eq!(items.len(), 1, "the nested real-PS5 layout must be found");
        match &items[0].launch {
            LaunchTarget::Game { path } => assert_eq!(
                path,
                &app.join("eboot.bin"),
                "must launch the eboot actually found, not <game>/eboot.bin"
            ),
            other => panic!("expected a Game launch target, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scan_dir_picks_up_a_conventional_cover_file() {
        let root = scratch_dir("with-cover");
        let game = root.join("nova-requiem");
        std::fs::create_dir_all(&game).unwrap();
        std::fs::write(game.join("eboot.bin"), b"stub").unwrap();
        std::fs::write(game.join("cover.png"), b"not decoded here - detection only").unwrap();

        let items = scan_dir(&root);
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0].cover_path.as_deref(),
            Some(game.join("cover.png").as_path())
        );
    }

    #[test]
    fn scan_dir_without_a_cover_file_leaves_cover_path_none() {
        let root = scratch_dir("no-cover");
        let game = root.join("kingfall");
        std::fs::create_dir_all(&game).unwrap();
        std::fs::write(game.join("eboot.bin"), b"stub").unwrap();

        let items = scan_dir(&root);
        assert_eq!(items.len(), 1);
        assert!(items[0].cover_path.is_none());
    }

    #[test]
    fn find_cover_prefers_png_over_jpg() {
        let root = scratch_dir("cover-priority");
        std::fs::write(root.join("cover.jpg"), b"jpg").unwrap();
        std::fs::write(root.join("cover.png"), b"png").unwrap();
        assert_eq!(find_cover(&root, None), Some(root.join("cover.png")));
    }

    #[test]
    fn find_cover_falls_back_to_the_titles_sce_sys_icon0() {
        // No user cover in the folder, but the title ships sce_sys/icon0.png
        // next to its eboot (the real PS5 layout) — that becomes the tile art.
        let root = scratch_dir("icon0-fallback");
        let app = root.join("PPSA00001-app");
        let sce_sys = app.join("sce_sys");
        std::fs::create_dir_all(&sce_sys).unwrap();
        let eboot = app.join("eboot.bin");
        std::fs::write(&eboot, b"stub").unwrap();
        let icon0 = sce_sys.join("icon0.png");
        std::fs::write(&icon0, b"png").unwrap();

        assert_eq!(find_cover(&root, Some(&eboot)), Some(icon0));
    }

    #[test]
    fn user_cover_wins_over_the_titles_icon0() {
        let root = scratch_dir("cover-over-icon0");
        std::fs::write(root.join("cover.png"), b"png").unwrap();
        let sce_sys = root.join("sce_sys");
        std::fs::create_dir_all(&sce_sys).unwrap();
        let eboot = root.join("eboot.bin");
        std::fs::write(&eboot, b"stub").unwrap();
        std::fs::write(sce_sys.join("icon0.png"), b"png").unwrap();

        assert_eq!(
            find_cover(&root, Some(&eboot)),
            Some(root.join("cover.png"))
        );
    }

    #[test]
    fn title_background_prefers_pic1_then_pic0() {
        let root = scratch_dir("bg-pick");
        let sce_sys = root.join("sce_sys");
        std::fs::create_dir_all(&sce_sys).unwrap();
        let eboot = root.join("eboot.bin");
        std::fs::write(&eboot, b"stub").unwrap();

        // Only pic0 present → pic0 (the boot splash doubles as key art).
        std::fs::write(sce_sys.join("pic0.png"), b"png").unwrap();
        assert_eq!(title_background(&eboot), Some(sce_sys.join("pic0.png")));

        // Dedicated pic1 present → it wins.
        std::fs::write(sce_sys.join("pic1.png"), b"png").unwrap();
        assert_eq!(title_background(&eboot), Some(sce_sys.join("pic1.png")));
    }
}
