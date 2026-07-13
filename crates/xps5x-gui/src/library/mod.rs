//! Library data model — the `LibraryItem`s the Shell renders.
//!
//! Decoupled from on-disk format (spec §4): items are discovered either by
//! [`scan`] (real game folders) or built in-code for built-in apps and the
//! SM0 sample library. All art is original — hand-picked gradients, never
//! Sony box-art (spec §11).

pub mod scan;

use egui::Color32;
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

    /// A neutral placeholder gradient for freshly-scanned titles that don't
    /// yet have real metadata (SM1 will attach real art).
    pub fn placeholder() -> Self {
        ArtSource::Game {
            hero: Gradient {
                hi: rgb(0x2b3a4e),
                mid: rgb(0x17222f),
                lo: rgb(0x0a1017),
            },
            tile: TileGradient {
                from: rgb(0x2b3a4e),
                to: rgb(0x17222f),
            },
        }
    }
}

/// One activity card (Continue / Trophies / Game Help / friends…).
#[derive(Debug, Clone)]
pub struct ActivityCard {
    pub top: String,
    pub main: String,
    pub sub: String,
    /// `Some(0..=100)` renders a progress bar (e.g. trophy completion).
    pub progress: Option<u8>,
}

/// Game-specific metadata shown in the Home context block.
#[derive(Debug, Clone)]
pub struct GameMeta {
    pub genre: String,
    pub players: String,
    /// Star rating, 0..=5.
    pub rating: u8,
    /// Kicker line ("Ready to play", "Continue — Chapter 4"…).
    pub kicker: String,
    pub activity: Vec<ActivityCard>,
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
                activity: vec![
                    ActivityCard { top: "Continue".to_string(), main: "Deep Reef Camp".to_string(), sub: "2d ago".to_string(), progress: None },
                    ActivityCard { top: "Trophies".to_string(), main: "15%".to_string(), sub: "6 / 40".to_string(), progress: Some(15) },
                    ActivityCard { top: "Game Help".to_string(), main: "Crafting guide".to_string(), sub: String::new(), progress: None },
                ],
            }),
            launch: LaunchTarget::Game { path: PathBuf::from("Games/tide") },
        },
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
}
