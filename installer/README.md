# XPS5X installer

Builds a sleek Windows setup wizard **and** a portable ZIP for XPS5X.

| Artifact | What it is |
|----------|-----------|
| `dist\XPS5X-<version>-Setup.exe` | Inno Setup wizard — per-user, non-elevated install |
| `dist\XPS5X-<version>-portable-win64.zip` | Extract-and-run; writes everything beside itself |

## Quick start

```powershell
# From the repo root (builds release, both artifacts):
installer\build.ps1
```

The script auto-detects the version from `Cargo.toml`, (re)generates the
branding art, runs `cargo build --release -p xps5x-gui`, packages the portable
ZIP, and — if Inno Setup 6 is installed — compiles the installer. Output lands
in `dist\`.

Useful switches: `-SkipBuild` (reuse `target\release\raeen.exe`),
`-SkipInstaller` (ZIP only), `-SkipPortable`, `-Version 0.2.0`.

## Prerequisites

- **Rust** toolchain (the workspace's pinned `rust-version`) — for the release build.
- **[Inno Setup 6](https://jrsoftware.org/isdl.php)** — for the installer step. Install with:
  ```powershell
  winget install --id JRSoftware.InnoSetup
  ```
  Without it, `build.ps1` still produces the portable ZIP and prints how to get Inno.
- **Windows PowerShell 5.1+** with `System.Drawing` (built in) — for branding art.

## What the installer does

1. **Welcome** → **License** (GPL-2.0, from `LICENSE`) → **install location**.
2. A custom **"Select your games folder"** page (defaults to
   `Documents\XPS5X\Games`). The folder is created and validated on *Next*.
3. **Tasks**: desktop icon, *start in a window instead of full-screen*.
4. Installs `raeen.exe`, the default theme, docs, and the app icon; creates
   `logs\`, `savedata\`, `shader_cache\`, `firmware\`.
5. If the **VC++ 2015-2022 x64 runtime** is missing and the redist is bundled
   (see below), installs it silently.
6. Writes a complete, hand-editable **`config.toml`** with
   `game_folders = ['<your chosen folder>']` (skipped if a `config.toml`
   already exists, so upgrades keep your settings).
7. Every shortcut is pinned to **`WorkingDir={app}`** and finish-page options
   let you launch XPS5X and/or open the games folder.

Uninstall removes the app but **keeps `savedata\` and `config.toml`**; only the
regenerable `shader_cache\` and `logs\` are cleared.

## Why per-user (non-elevated)?

The app reads and writes everything — `config.toml`, `logs\`, `savedata\`,
`shader_cache\`, `themes\`, the games scan — **relative to its working
directory**. Installing per-user into `%LocalAppData%\Programs\XPS5X` (writable,
no admin) with `WorkingDir={app}` on every shortcut keeps all of that working.

An **all-users / Program Files** install would put those writes in a
non-writable location. That's a ~10-line app change (redirect user-data to
`%LocalAppData%\XPS5X`), deliberately left as a follow-up — see below.

## Bundling the Visual C++ runtime (optional but recommended)

XPS5X is a standard MSVC Rust build, so target PCs need the **Microsoft Visual
C++ 2015-2022 x64 Redistributable**. Most systems already have it. To bundle it
so the installer can fix machines that don't:

1. Download `vc_redist.x64.exe` from
   <https://aka.ms/vs/17/release/vc_redist.x64.exe>.
2. Save it to **`installer\redist\vc_redist.x64.exe`**.

That's it — `xps5x.iss` detects the file at compile time and installs it
silently only when the runtime is actually missing. Without the file, the step
is compiled out and setup still works (it just assumes the runtime is present).

## Files

```
installer/
├── xps5x.iss             Inno Setup script (the wizard)
├── build.ps1             build + package both artifacts
├── README.md             this file
├── assets/               generated branding (ico + wizard bmps)
├── scripts/
│   └── gen-assets.ps1    regenerable branding generator (System.Drawing)
└── redist/               drop vc_redist.x64.exe here to bundle it
```

### Regenerating / rebranding the art

```powershell
installer\scripts\gen-assets.ps1
```

Edit the palette / `$GLYPH` constants at the top of `gen-assets.ps1` to
recolor or restyle the badge, wordmark, and wizard images.

## Follow-ups (not done here)

- **Embed the icon in `raeen.exe`** (via a `build.rs` + `winres`/`embed-resource`)
  and set the eframe window icon, so the bare exe and the running taskbar show
  the XPS5X mark. Today the branded icon reaches shortcuts and Add/Remove
  Programs via the shipped `xps5x.ico`.
- **All-users install support**: redirect user-data to `%LocalAppData%\XPS5X`
  so a Program Files install can write config/logs/saves.
- **Code signing**: sign `Setup.exe` and `raeen.exe` to drop the SmartScreen
  warning. `build.ps1` has an obvious spot to add `signtool` if you get a cert.
