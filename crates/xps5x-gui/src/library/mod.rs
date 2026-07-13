//! Library data model — the `LibraryItem`s the Shell renders.
//!
//! Decoupled from on-disk format (spec §4): items are discovered either by
//! [`scan`] (real game folders) or built in-code for built-in apps and the
//! SM0 sample library. All art is original — hand-picked gradients, never
//! Sony box-art (spec §11).

pub mod scan;

use egui::Color32;
use std::collections::HashMap;
use std::path::PathBuf;

/// Parse a `0xRRGGBB` literal into a [`Color32`] at full opacity.
const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb(
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    )
}

/// Whether a [`LibraryItem`] is a playable game or a built-in Shell app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Game,
    App,
}

/// A three-stop diagonal gradient used for hero backgrounds.
#[derive(Debug, Clone, Copy)]
pub struct Gradient {
    /// Bright accent stop (upper area of the art).
    pub hi: Color32,
    /// Mid transition stop.
    pub mid: Color32,
    /// Dark base stop (matches the ground color family).
    pub lo: Color32,
}

/// A simple two-stop diagonal gradient used for rail tiles.
#[derive(Debug, Clone, Copy)]
pub struct TileGradient {
    pub from: Color32,
    pub to: Color32,
}

/// Original vector glyph drawn for built-in apps (no Sony iconography).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphKind {
    Bag,
    Grid,
    Gear,
    /// Media tab (spec §10 SM2): music app.
    Music,
    /// Media tab (spec §10 SM2): video app.
    Video,
    /// Media tab (spec §10 SM2): web browser app.
    Network,
}

/// Where a [`LibraryItem`]'s art comes from.
#[derive(Debug, Clone)]
pub enum ArtSource {
    Game { hero: Gradient, tile: TileGradient },
    App { tile: TileGradient, glyph: GlyphKind },
}

impl ArtSource {
    pub fn tile(&self) -> TileGradient {
        match self {
            ArtSource::Game { tile, .. } => *tile,
            ArtSource::App { tile, .. } => *tile,
        }
    }

    /// The hero background gradient. Apps don't have dedicated hero art, so
    /// one is derived from their tile colors.
    pub fn hero(&self) -> Gradient {
        match self {
            ArtSource::Game { hero, .. } => *hero,
            ArtSource::App { tile, .. } => Gradient { hi: tile.from, mid: tile.to, lo: rgb(0x0a1017) },
        }
    }

}

/// Blend two color channels, `t` in `0.0..=1.0`.
fn lerp_channel(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    Color32::from_rgb(lerp_channel(a.r(), b.r(), t), lerp_channel(a.g(), b.g(), t), lerp_channel(a.b(), b.b(), t))
}

/// Build gradient art (hero + tile) from two explicit stops — a bright
/// highlight and a dark base. Used both for `xps5x-title.toml`'s optional
/// `gradient` field and for id-derived art (spec §4: gradients only, no
/// image decoding in SM1).
pub fn art_from_stops(hi: Color32, lo: Color32) -> ArtSource {
    let mid = lerp_color(hi, lo, 0.5);
    ArtSource::Game {
        hero: Gradient { hi, mid, lo },
        tile: TileGradient { from: hi, to: lo },
    }
}

/// Deterministically derive gradient art from a stable id string — used
/// when a game has no `xps5x-title.toml`, or one without an explicit
/// `gradient`. Pure hash + HSL math, no image assets or new dependencies
/// (spec §11).
pub fn art_from_id(id: &str) -> ArtSource {
    let hue = (fnv1a(id) % 360) as f32;
    let hi = hsl_to_rgb(hue, 0.68, 0.60);
    let lo = hsl_to_rgb(hue, 0.55, 0.09);
    art_from_stops(hi, lo)
}

/// FNV-1a 32-bit hash — small, dependency-free, deterministic.
fn fnv1a(s: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in s.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Minimal HSL → RGB conversion (`h` in degrees `0..360`, `s`/`l` in
/// `0.0..=1.0`).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color32 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    Color32::from_rgb(
        ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

/// In-memory cache of parsed per-game metadata, keyed by [`LibraryItem::id`]
/// (spec §4). The Shell builds one from the scanned/sample library and Home
/// reads a focused item's metadata from it, decoupling rendering from how
/// (or whether) the metadata was sourced.
#[derive(Debug, Clone, Default)]
pub struct MetaCache {
    entries: HashMap<String, GameMeta>,
}

impl MetaCache {
    /// Reserved for callers that build a cache incrementally (e.g. a future
    /// async/background scan) rather than from a finished item list.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a cache from a set of items' already-embedded metadata. Works
    /// uniformly for a scanned library and the in-code sample library.
    pub fn from_items(items: &[LibraryItem]) -> Self {
        let mut entries = HashMap::new();
        for item in items {
            if let Some(meta) = &item.meta {
                entries.insert(item.id.clone(), meta.clone());
            }
        }
        Self { entries }
    }

    pub fn get(&self, id: &str) -> Option<&GameMeta> {
        self.entries.get(id)
    }

    /// Reserved for callers that update a single game's metadata in place
    /// (e.g. a future Settings "rescan" action) without rebuilding the
    /// whole cache via [`MetaCache::from_items`].
    #[allow(dead_code)]
    pub fn insert(&mut self, id: String, meta: GameMeta) {
        self.entries.insert(id, meta);
    }

    /// Reserved for callers (e.g. a future Settings library summary) that
    /// need the count without iterating.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Reserved for callers that need an early-out before iterating (no
    /// current call site iterates the cache directly).
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One activity card (Continue / Trophies / Game Help / friends…). Parsed
/// from `xps5x-title.toml`; the concept-style Home renders only the trophy
/// progress (via [`GameMeta::progress_percent`]) — the text fields are kept
/// for a future detail/library view.
#[derive(Debug, Clone)]
pub struct ActivityCard {
    #[allow(dead_code)]
    pub top: String,
    #[allow(dead_code)]
    pub main: String,
    #[allow(dead_code)]
    pub sub: String,
    /// `Some(0..=100)` — trophy/completion progress.
    pub progress: Option<u8>,
}

/// Game-specific metadata shown in the Home context block.
#[derive(Debug, Clone)]
pub struct GameMeta {
    /// Parsed from `xps5x-title.toml` but not rendered by the concept-style
    /// Home (title + play stats only) — kept for a future detail view.
    #[allow(dead_code)]
    pub genre: String,
    /// Same parsed-but-not-rendered status as `genre`.
    #[allow(dead_code)]
    pub players: String,
    /// Star rating, 0..=5. Same parsed-but-not-rendered status as `genre`.
    #[allow(dead_code)]
    pub rating: u8,
    /// Kicker line ("Ready to play", "Continue — Chapter 4"…). Same
    /// parsed-but-not-rendered status as `genre`.
    #[allow(dead_code)]
    pub kicker: String,
    /// Total play time as display text ("2h 1m") — the Home "Time played"
    /// stat. Empty when unknown.
    pub time_played: String,
    /// Most recently earned trophy name — the Home "Last trophy" stat.
    /// Empty when unknown.
    pub last_trophy: String,
    pub activity: Vec<ActivityCard>,
}

impl GameMeta {
    /// Trophy/completion percent for the Home "Progress" stat: the first
    /// activity card that carries a progress bar (the Trophies card, by
    /// convention).
    pub fn progress_percent(&self) -> Option<u8> {
        self.activity.iter().find_map(|card| card.progress)
    }
}

/// Where the engine should be pointed to launch this item.
///
/// `StubLauncher` ignores the payload; the real engine launcher (SM3) reads
/// it to actually start a title, which is when these fields get read.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum LaunchTarget {
    Game { path: PathBuf },
    App { id: String },
}

/// A single tile in the Home rail: a game or a built-in app.
#[derive(Debug, Clone)]
pub struct LibraryItem {
    /// Stable identity, used for save-state/session bookkeeping once the
    /// engine seam (SM3) needs to correlate library entries to sessions.
    #[allow(dead_code)]
    pub id: String,
    pub title: String,
    pub kind: ItemKind,
    pub art: ArtSource,
    pub meta: Option<GameMeta>,
    pub launch: LaunchTarget,
}

/// The Shell's built-in apps (Store, Game Library, Settings) — always
/// present on the Games rail regardless of whether the library came from a
/// real scan or the sample fallback, so Settings (and, later, Store/
/// Library) stay reachable even on a fresh checkout with real games but no
/// prior apps baked in (spec §10 SM2: Settings must be reachable from its
/// Home rail tile).
pub fn built_in_apps() -> Vec<LibraryItem> {
    vec![
        LibraryItem {
            id: "store".to_string(),
            title: "Store".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient { from: rgb(0x1f8fff), to: rgb(0x0a4bc2) },
                glyph: GlyphKind::Bag,
            },
            meta: None,
            launch: LaunchTarget::App { id: "store".to_string() },
        },
        LibraryItem {
            id: "library".to_string(),
            title: "Game Library".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient { from: rgb(0x2b3a4e), to: rgb(0x17222f) },
                glyph: GlyphKind::Grid,
            },
            meta: None,
            launch: LaunchTarget::App { id: "library".to_string() },
        },
        LibraryItem {
            id: "settings".to_string(),
            title: "Settings".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient { from: rgb(0x2b3a4e), to: rgb(0x17222f) },
                glyph: GlyphKind::Gear,
            },
            meta: None,
            launch: LaunchTarget::App { id: "settings".to_string() },
        },
    ]
}

/// The mockup's sample library — original invented titles and gradient art,
/// used until real game-folder scanning (SM1) is wired to the Shell.
pub fn sample_library() -> Vec<LibraryItem> {
    vec![
        LibraryItem {
            id: "nova".to_string(),
            title: "Nova Requiem".to_string(),
            kind: ItemKind::Game,
            art: ArtSource::Game {
                hero: Gradient { hi: rgb(0xff4d6d), mid: rgb(0x7a1338), lo: rgb(0x16060f) },
                tile: TileGradient { from: rgb(0xff5a7a), to: rgb(0x240712) },
            },
            meta: Some(GameMeta {
                genre: "Action RPG".to_string(),
                players: "Single-player".to_string(),
                rating: 5,
                kicker: "Ready to play".to_string(),
                time_played: "2h 1m".to_string(),
                last_trophy: "Hollow Walker".to_string(),
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "Chapter 4 — The Hollow".to_string(), sub: "2h ago".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "58%".to_string(), sub: "24 / 41".to_string(), progress: Some(58) },
                    ActivityCard { top: "Game Help".to_string(), main: "3 tips available".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/nova") },
        },
        LibraryItem {
            id: "astral".to_string(),
            title: "Astral Drift".to_string(),
            kind: ItemKind::Game,
            art: ArtSource::Game {
                hero: Gradient { hi: rgb(0x2fe0d0), mid: rgb(0x0e6a8c), lo: rgb(0x05131f) },
                tile: TileGradient { from: rgb(0x3ff0dc), to: rgb(0x04202f) },
            },
            meta: Some(GameMeta {
                genre: "Open World".to_string(),
                players: "Online Co-op".to_string(),
                rating: 4,
                kicker: "Continue — Sector 12".to_string(),
                time_played: "31h 6m".to_string(),
                last_trophy: "Drift Racer".to_string(),
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "Sector 12 — Drift Run".to_string(), sub: "Yesterday".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "31%".to_string(), sub: "12 / 38".to_string(), progress: Some(31) },
                    ActivityCard { top: "2 friends playing".to_string(), main: "Join session".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/astral") },
        },
        LibraryItem {
            id: "sable".to_string(),
            title: "Sable Horizon".to_string(),
            kind: ItemKind::Game,
            art: ArtSource::Game {
                hero: Gradient { hi: rgb(0xffb454), mid: rgb(0xc25a1a), lo: rgb(0x2a1207) },
                tile: TileGradient { from: rgb(0xffc06a), to: rgb(0x351808) },
            },
            meta: Some(GameMeta {
                genre: "Adventure".to_string(),
                players: "Single-player".to_string(),
                rating: 5,
                kicker: "Ready to play".to_string(),
                time_played: "104h 22m".to_string(),
                last_trophy: "Dune Strider".to_string(),
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "The Dunes".to_string(), sub: "3d ago".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "92%".to_string(), sub: "33 / 36".to_string(), progress: Some(92) },
                    ActivityCard { top: "Game Help".to_string(), main: "1 tip available".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/sable") },
        },
        LibraryItem {
            id: "kingfall".to_string(),
            title: "Kingfall".to_string(),
            kind: ItemKind::Game,
            art: ArtSource::Game {
                hero: Gradient { hi: rgb(0x7bd88f), mid: rgb(0x1f6b45), lo: rgb(0x071710) },
                tile: TileGradient { from: rgb(0x8fe6a2), to: rgb(0x082115) },
            },
            meta: Some(GameMeta {
                genre: "Soulslike".to_string(),
                players: "Single-player".to_string(),
                rating: 4,
                kicker: "Continue — Ashen Keep".to_string(),
                time_played: "12h 40m".to_string(),
                last_trophy: "Ashen Victor".to_string(),
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "Ashen Keep — Boss".to_string(), sub: "1h ago".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "44%".to_string(), sub: "18 / 41".to_string(), progress: Some(44) },
                    ActivityCard { top: "Game Help".to_string(), main: "Boss strategy".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/kingfall") },
        },
        LibraryItem {
            id: "neon".to_string(),
            title: "Neon Verge".to_string(),
            kind: ItemKind::Game,
            art: ArtSource::Game {
                hero: Gradient { hi: rgb(0xff5cf0), mid: rgb(0x6a1bd6), lo: rgb(0x100626) },
                tile: TileGradient { from: rgb(0xff6cf2), to: rgb(0x150931) },
            },
            meta: Some(GameMeta {
                genre: "Cyberpunk FPS".to_string(),
                players: "Multiplayer".to_string(),
                rating: 4,
                kicker: "Ready to play".to_string(),
                time_played: "5h 12m".to_string(),
                last_trophy: "Division Climber".to_string(),
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "Ranked — Div 2".to_string(), sub: "5h ago".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "27%".to_string(), sub: "9 / 33".to_string(), progress: Some(27) },
                    ActivityCard { top: "5 friends online".to_string(), main: "Invite to party".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/neon") },
        },
        LibraryItem {
            id: "tide".to_string(),
            title: "Tidewrought".to_string(),
            kind: ItemKind::Game,
            art: ArtSource::Game {
                hero: Gradient { hi: rgb(0x4aa8ff), mid: rgb(0x17457e), lo: rgb(0x050f1f) },
                tile: TileGradient { from: rgb(0x63b6ff), to: rgb(0x061a30) },
            },
            meta: Some(GameMeta {
                genre: "Survival".to_string(),
                players: "Online Co-op".to_string(),
                rating: 4,
                kicker: "Ready to play".to_string(),
                time_played: "48m".to_string(),
                last_trophy: "First Dive".to_string(),
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "Deep Reef Camp".to_string(), sub: "2d ago".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "15%".to_string(), sub: "6 / 40".to_string(), progress: Some(15) },
                    ActivityCard { top: "Game Help".to_string(), main: "Crafting guide".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/tide") },
        },
    ]
    .into_iter()
    .chain(built_in_apps())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_library_matches_mockup_shape() {
        let items = sample_library();
        assert_eq!(items.len(), 9);
        assert_eq!(items[0].title, "Nova Requiem");
        assert_eq!(items.iter().filter(|i| i.kind == ItemKind::Game).count(), 6);
        assert_eq!(items.iter().filter(|i| i.kind == ItemKind::App).count(), 3);
    }

    #[test]
    fn built_in_apps_always_includes_a_settings_tile() {
        let apps = built_in_apps();
        assert_eq!(apps.len(), 3);
        assert!(apps.iter().all(|i| i.kind == ItemKind::App));
        assert!(apps.iter().any(|i| i.id == "settings"), "Settings must always be reachable on the Games rail");
    }

    #[test]
    fn sample_library_ends_with_the_built_in_apps() {
        let items = sample_library();
        let apps = built_in_apps();
        let tail_ids: Vec<&str> = items[items.len() - apps.len()..].iter().map(|i| i.id.as_str()).collect();
        let app_ids: Vec<&str> = apps.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(tail_ids, app_ids);
    }

    #[test]
    fn art_from_id_is_deterministic() {
        let a = art_from_id("nova-requiem").tile();
        let b = art_from_id("nova-requiem").tile();
        assert_eq!(a.from, b.from);
        assert_eq!(a.to, b.to);
    }

    #[test]
    fn art_from_id_varies_across_ids() {
        let a = art_from_id("nova-requiem").tile();
        let b = art_from_id("sable-horizon").tile();
        assert!(a.from != b.from || a.to != b.to);
    }

    #[test]
    fn art_from_stops_uses_exact_stops() {
        let hi = rgb(0xff0000);
        let lo = rgb(0x000000);
        let art = art_from_stops(hi, lo);
        let tile = art.tile();
        assert_eq!(tile.from, hi);
        assert_eq!(tile.to, lo);
        let hero = art.hero();
        assert_eq!(hero.hi, hi);
        assert_eq!(hero.lo, lo);
    }

    #[test]
    fn meta_cache_from_items_only_keeps_games_with_meta() {
        let items = sample_library();
        let cache = MetaCache::from_items(&items);
        // 6 games in the sample library, all with meta; 3 apps, none.
        assert_eq!(cache.len(), 6);
        assert!(cache.get("nova").is_some());
        assert!(cache.get("store").is_none());
        assert!(cache.get("does-not-exist").is_none());
    }
}
