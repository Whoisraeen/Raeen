# Raeen Shell — Design Spec

**Status:** Approved direction (visual reference locked via interactive mockup)
**Date:** 2026-07-13
**Crate:** `raeen-gui`
**Companion doc:** `docs/superpowers/specs/2026-07-12-raeen-lle-firmware-spine-design.md` (the engine)

---

## 1. Goal

Raeen launches into a **full-screen, PS5-style console experience** — its own dashboard ("the Shell") — that boots, lists the user's games, and hands a chosen title to the emulation engine to run with enhancements (unlocked framerate, higher resolution). To the person on the couch, **the app is a console**. The Shell is Raeen's own software; it is not Sony's `SceShellUI`.

This spec covers the Shell only: what the user sees and operates. The engine that actually runs a game (loader → kernel → module layer → GPU translation) is specified separately.

## 2. What "PS5-like" means here (and the hard IP line)

The Shell recreates the **layout, motion language, and interaction model** of the PS5 home — none of which are copyrightable. It does **not** ship Sony's copyrighted assets: the SST font, Sony's icon artwork, sound effects, dynamic background art, logos/wordmarks, or any game box-art. Those are replaced with original Raeen assets.

**Authenticity path — theming.** The Shell is asset-driven through a theme layer (§6). Raeen ships a **default original theme**. A user may install a **local theme** built from assets they extracted from firmware/hardware they own. That local theme never ships with Raeen and is never committed to the repo. This keeps the distributable clean while allowing a personal build to look exactly like the real thing.

**Non-negotiable:** the repository and any distributable binary contain zero Sony assets, keys, or firmware.

## 3. Screens & navigation model

The reference is the interactive mockup (`scratchpad/raeen-shell.html`). Screens, in build priority:

1. **Boot** — brief power-on sequence (logo + progress shimmer), fades into Home.
2. **Home** — the primary screen:
   - **Function bar** (top): Games / Media tabs (left); Search, Friends, Notifications, Settings, Profile avatar, and clock (right).
   - **Hero** (background): full-bleed art for the focused title, with a legibility scrim.
   - **Context block**: focused title's wordmark, rating/genre/players meta, **Play** (primary) + More (secondary), and a row of **activity cards** (Continue / Trophies-with-progress / Game Help or friends).
   - **Tile rail**: horizontal row of game/app tiles. The focused tile holds a fixed anchor (left third); the rail slides beneath it. Focused tile scales up with a white ring and floating label.
3. **Control Center** — bottom overlay summoned by the PS/Guide button (mockup: `C`). A row of cards — Home, Switcher, Notifications, Game Base, Music, Sound, Microphone, Accessories, Profile, Network, Power — with the focused card expanding a summary panel above it. Dims the screen behind.
4. **Media tab** — same rail model, media apps instead of games (later milestone).
5. **Settings** — system settings (video, audio, theme selection, game folders, key provider path) (later milestone).

**Input model:** designed D-pad/stick-first (like a console), with keyboard and mouse as equals on desktop.
- Left/Right: move rail focus. Up/Down: move between rail, activity cards, function bar.
- Confirm (Enter / gamepad South): launch focused title or activate control.
- Guide (gamepad PS button / `C`): toggle Control Center.
- Back (Esc / gamepad East): close overlay / go up a level.

## 4. Content model — the game library

- The Shell scans one or more **user game folders** (configured in Settings; default `./Games`).
- Each installed title is discovered from its on-disk package/metadata. Title, art, and metadata come from the package plus a local metadata cache; **no online store calls** in the base design.
- Apps (Store, Library, Settings) are built-in Shell entries, not games.
- The library is presented as `LibraryItem`s to the rendering layer, decoupled from disk format.

```rust
pub struct LibraryItem {
    pub id: String,               // stable id (e.g. title id)
    pub title: String,
    pub kind: ItemKind,           // Game | App
    pub art: ArtSource,           // themed placeholder or user-provided art
    pub meta: Option<GameMeta>,   // genre, players, rating, activity
    pub launch: LaunchTarget,     // path/handle the engine consumes
}
```

## 5. Shell ↔ Engine interface

The Shell never contains emulation logic. It calls into the engine through one narrow trait, so the Shell can be developed and tested against a stub before the engine can run anything.

```rust
/// Implemented by the engine; consumed by the Shell.
pub trait GameLauncher {
    /// Begin launching a title. Returns a handle the Shell polls for state.
    fn launch(&self, target: &LaunchTarget) -> Result<SessionHandle, LaunchError>;
    /// Current state of a running session (Loading, Running, Faulted, Exited).
    fn session_state(&self, handle: &SessionHandle) -> SessionState;
    /// Request a running session to quit (returns to Shell).
    fn quit(&self, handle: &SessionHandle) -> Result<(), LaunchError>;
}
```

- On **Play**, the Shell shows a launch transition, calls `launch`, and on `Running` yields the display to the engine's output surface.
- On session exit/fault, control returns to the Shell (Home, same focused tile).
- For now the Shell ships with a `StubLauncher` that simulates Loading→Running→Exited, so the whole Shell is exercisable end-to-end without the engine.

## 6. Theming system (the authenticity seam)

A `Theme` is a directory of assets + a manifest resolved at runtime.

```
themes/
  default/            # ships with Raeen — original assets only
    theme.toml        # palette, type, sounds, layout tokens
    fonts/…  icons/…  sounds/…  backgrounds/…
  <user-theme>/       # user-installed, NOT in repo, gitignored location
```

- `theme.toml` defines palette tokens (ground, raised, accent, focus…), the type face(s), icon set, sound cues, and layout metrics — the same tokens the mockup encodes as CSS custom properties.
- The renderer reads **only** tokens/asset handles from the active theme; no colors or asset paths are hard-coded in widgets.
- Theme selection lives in Settings. Missing assets fall back to `default`.
- **Guardrail:** the loader treats user themes as untrusted content — bounds-checked, no code execution, images decoded through safe decoders.

## 7. Tech stack

The workspace already commits to **`eframe` + `egui` on the `wgpu` backend**, with `winit` and `gilrs` (gamepad) present. Decision: **build the Shell in `egui`** now.

- **Why:** it's the established stack (honor existing patterns), it's pure-Rust and GPU-accelerated via `wgpu`, and it gets a *running native Shell* fastest — the point of this milestone is to see it real.
- **Animations:** egui is immediate-mode; the mockup's crossfades, eased rail slides, and focus scaling are driven by an explicit `Animator` (time-based lerps via `ctx.animate_*` / a small tween helper), isolated so widgets stay declarative.
- **Isolation for the future:** all rendering sits behind a `ShellRenderer` boundary and all art/color behind the theme layer. When the Shell must later **composite over live game output** (the engine's Vulkan surface), the renderer can migrate to a `wgpu`-native compositor without touching Shell logic, navigation, or the library/launcher layers.
- **Alternative considered:** Slint (declarative, nicer built-in animation; used by Obliteration). Rejected for now only to avoid a second UI framework diverging from the existing egui code; revisit if egui animation friction proves high.

## 8. Module structure (`raeen-gui`)

```
src/
  main.rs            # entry; keeps --firmware-info; launches Shell
  app.rs             # eframe App impl; owns Shell state, drives frames
  shell/
    mod.rs           # Shell state machine (Boot→Home→…), input routing
    home.rs          # Home screen (function bar, hero, context, rail)
    control_center.rs# Control Center overlay
    boot.rs          # boot sequence
    nav.rs           # focus model + input mapping (keyboard + gamepad)
    anim.rs          # time-based tween/animator helpers
  library/
    mod.rs           # LibraryItem model
    scan.rs          # game-folder scanning + metadata cache
  theme/
    mod.rs           # Theme, token resolution, asset handles
    loader.rs        # load/validate a theme directory (untrusted-safe)
  launcher.rs        # GameLauncher trait + StubLauncher
```

## 9. Testing

- **library/scan:** unit tests over synthetic game folders (fixtures) → expected `LibraryItem`s; malformed entries are skipped, not fatal.
- **theme/loader:** valid theme loads; missing assets fall back to default; malformed/oversized/hostile files rejected cleanly (no panic).
- **nav:** focus transitions are pure functions over (state, input) → state; table-driven tests for the full navigation graph.
- **launcher:** `StubLauncher` drives Loading→Running→Exited; Shell returns to Home on exit.
- **Manual/visual:** a `--shell-demo` run boots the native Shell with the stub launcher and sample library for eyeball parity against the mockup.

## 10. Milestones

- **SM0 — Native Home shell.** egui Shell: boot → Home with function bar, hero, context block, tile rail, focus navigation (keyboard + gamepad), theme layer with the default theme, `StubLauncher`, sample library. Deliverable: `cargo run` opens a full-screen PS5-style Home you can navigate and "launch" (stub). Matches the mockup.
- **SM1 — Control Center + library scan + activity cards.** Real game-folder scanning + metadata cache; activity cards; Control Center overlay with functional panels (Sound/Network/Power do real things where meaningful).
- **SM2 — Media + Settings + theme install.** Media tab, Settings (video/audio/game folders/key-provider path/theme selection), user-theme install flow.
- **SM3 — Engine handoff.** Replace `StubLauncher` with the real engine `GameLauncher`; launch transition yields to the engine surface; return-to-Shell on exit.

## 11. Global constraints

- Rust edition 2024, rust-version ≥ 1.85; GPL-2.0-only.
- **Zero Sony assets/keys/firmware** in repo or distributable. Original assets only in `themes/default`. User themes are gitignored and never committed.
- Theme loader treats all user-supplied assets as untrusted (bounds-checked, no code execution).
- Shell contains no emulation logic; it talks to the engine only through `GameLauncher`.
- Honor existing workspace patterns (egui/eframe/wgpu/winit/gilrs); no new UI framework without cause.
```
