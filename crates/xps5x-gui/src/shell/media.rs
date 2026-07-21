//! Media tab rail — original XPS5X media apps (spec §10 SM2).
//!
//! The Media tab reuses the same rail rendering and focus model as Games
//! (`home::draw`/`home::draw_rail` and `nav::NavState`'s `rail_index`); these
//! are ordinary [`LibraryItem`]s of [`ItemKind::App`] with distinct,
//! original names, gradients, and icons — never Sony's Music/Video/Browser
//! app branding (spec §11). Confirming one is a no-op/log for now: there is
//! no media-playback engine to hand off to (spec §5 keeps the Shell itself
//! free of any such logic).

use crate::library::{ArtSource, GlyphKind, ItemKind, LaunchTarget, LibraryItem, TileGradient};
use egui::Color32;

/// Parse a `0xRRGGBB` literal into a [`Color32`] at full opacity. Small
/// helper duplicated per-module (see `library::rgb`, `theme::rgb`) rather
/// than threading a shared one across module boundaries for three lines.
const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb(
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    )
}

/// The Media tab's fixed rail. Ids are namespaced (`media:*`) so they can
/// never collide with a Games-rail id.
pub fn media_items() -> Vec<LibraryItem> {
    vec![
        LibraryItem {
            id: "media:wavelength".to_string(),
            title: "Wavelength".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient {
                    from: rgb(0x3ff0a0),
                    to: rgb(0x04241a),
                },
                glyph: GlyphKind::Music,
            },
            meta: None,
            cover_path: None,
            title_id: None,
            version: None,
            launch: LaunchTarget::App {
                id: "media:wavelength".to_string(),
            },
        },
        LibraryItem {
            id: "media:screening-room".to_string(),
            title: "Screening Room".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient {
                    from: rgb(0xff9a4d),
                    to: rgb(0x2a1408),
                },
                glyph: GlyphKind::Video,
            },
            meta: None,
            cover_path: None,
            title_id: None,
            version: None,
            launch: LaunchTarget::App {
                id: "media:screening-room".to_string(),
            },
        },
        LibraryItem {
            id: "media:wayfinder".to_string(),
            title: "Wayfinder".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient {
                    from: rgb(0x4aa8ff),
                    to: rgb(0x061a30),
                },
                glyph: GlyphKind::Network,
            },
            meta: None,
            cover_path: None,
            title_id: None,
            version: None,
            launch: LaunchTarget::App {
                id: "media:wayfinder".to_string(),
            },
        },
        LibraryItem {
            id: "media:snapshots".to_string(),
            title: "Snapshots".to_string(),
            kind: ItemKind::App,
            art: ArtSource::App {
                tile: TileGradient {
                    from: rgb(0xff5cf0),
                    to: rgb(0x150931),
                },
                glyph: GlyphKind::Grid,
            },
            meta: None,
            cover_path: None,
            title_id: None,
            version: None,
            launch: LaunchTarget::App {
                id: "media:snapshots".to_string(),
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_items_are_all_apps_with_original_titles() {
        let items = media_items();
        assert_eq!(items.len(), 4);
        for item in &items {
            assert_eq!(item.kind, ItemKind::App);
            assert!(item.meta.is_none());
            assert!(matches!(item.launch, LaunchTarget::App { .. }));
        }
        let titles: Vec<&str> = items.iter().map(|i| i.title.as_str()).collect();
        assert_eq!(
            titles,
            ["Wavelength", "Screening Room", "Wayfinder", "Snapshots"]
        );
    }

    #[test]
    fn media_item_ids_are_unique_and_namespaced() {
        let items = media_items();
        let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
        for id in &ids {
            assert!(id.starts_with("media:"), "id should be namespaced: {id}");
        }
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "ids must be unique");
    }
}
