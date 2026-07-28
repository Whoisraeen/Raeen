# Raeen — PlayStation 5 Emulator

<p align="center">
  <strong>A cross-platform PS5 compatibility layer that translates PS5 system calls and GPU commands to run natively on your PC hardware.</strong>
</p>

<p align="center">
  <a href="https://github.com/Whoisraeen/Raeen/actions/workflows/ci.yml"><img src="https://github.com/Whoisraeen/Raeen/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Whoisraeen/Raeen/releases/latest"><img src="https://img.shields.io/github/v/release/Whoisraeen/Raeen?include_prereleases&label=download" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--2.0--only-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://whoisraeen.github.io/raeen-website/"><strong>Website</strong></a> ·
  <a href="https://github.com/Whoisraeen/Raeen/releases/latest">Download (Windows x64)</a> ·
  <a href="https://github.com/Whoisraeen/Raeen/issues">Issues</a>
</p>

> **Status: early alpha — one title renders, the rest do not.**
>
> The honest picture, measured rather than claimed:
>
> - **Minecraft Bedrock** (user-owned retail) reaches an **interactive 3D world
>   with textures, HUD and working save data**, at ~56 FPS on the reference
>   machine. This is the one title that genuinely plays.
> - **GTA V, ASTRO.BOT, Avatar: Frontiers of Pandora** boot, link every import,
>   and present frames, but do **not** render a recognizable image yet. They stop
>   on specific, named shader-translation gaps.
> - **Until Dawn, Subnautica: Below Zero, Dragon Ball Sparking Zero** launch and
>   run without crashing but present **no frames**.
> - **A Plague Tale Requiem** crashes, with a full diagnostic report.
>
> Every one of those statements comes from an automated per-title measurement
> (`cargo xtask baseline run`), and the raw table — including each title's exact
> first blocker — is in **[compat/COMPATIBILITY.md](compat/COMPATIBILITY.md)**.
> "Recognizable frames" is not the same as "correct rendering", and none of this
> is a playability or stability claim beyond the one title named above.
>
> Releases include a built-in auto-updater (Settings → System).
>
> **Development note:** Raeen is built with heavy AI assistance; commits carry a
> `Co-Authored-By` trailer recording it. Design decisions, measurements and
> acceptance criteria are reviewed by a human before they land.

---

## What is Raeen?

Raeen is a PS5 emulator / compatibility layer that enables PS5 game binaries to run on Windows, Linux, and macOS. Since the PS5 uses an x86-64 CPU (AMD Zen 2), Raeen **natively executes** game code on your PC — no slow interpretation needed. The magic happens in the **translation layers**:

- **Kernel HLE** — Translates PS5's Orbis OS (FreeBSD-based) system calls to your host OS
- **GPU Translation** — Converts Sony's GNM/PM4 GPU commands to Vulkan (or Metal on macOS)
- **Shader Recompiler** — Recompiles RDNA2 ISA shader binaries to SPIR-V in real-time
- **Hardware Emulation** — Emulates Tempest 3D Audio, the custom I/O complex, and DualSense features

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raeen Application                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Game     │  │ Settings │  │  Debug   │  │  Log    │ │
│  │  Library  │  │  Panel   │  │  Tools   │  │  Viewer │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Binary Loader                          │
│            (SELF / ELF / PKG Parser)                     │
├─────────────────────────────────────────────────────────┤
│              Orbis Kernel HLE Layer                      │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐ │
│  │ Syscall │ │  Memory  │ │ Thread  │ │  Virtual FS  │ │
│  │ Dispatch│ │ Manager  │ │ Sched.  │ │              │ │
│  └─────────┘ └──────────┘ └─────────┘ └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│           Hardware Translation Layer                     │
│  ┌───────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ GPU: GNM→VK   │  │  Shader  │  │ Tempest Audio    │ │
│  │ PM4 Decoder   │  │ Recomp.  │  │ 3D Spatial       │ │
│  │ Vulkan Backend│  │ ISA→SPIRV│  │                  │ │
│  └───────────────┘  └──────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────┤
│              Host Platform (Vulkan 1.3)                  │
│         Windows  ·  Linux  ·  macOS (Metal)              │
└─────────────────────────────────────────────────────────┘
```

## Building

### Prerequisites
- Rust 1.85+ (install via [rustup](https://rustup.rs))
- Vulkan SDK 1.3+ (install from [LunarG](https://vulkan.lunarg.com/sdk/home))
- CMake 3.20+ (for some native dependencies)

### Build
```bash
# Clone the repository
git clone https://github.com/Whoisraeen/Raeen.git
cd Raeen

# Build in release mode
cargo build --release

# Run
cargo run --release
```

## Project Structure

```
crates/
├── raeen-core/      Core engine, configuration, error handling
├── raeen-loader/    SELF/ELF/PKG binary parser and loader
├── raeen-kernel/    Orbis OS kernel HLE (syscall translation)
├── raeen-gpu/       GPU command translation (GNM→Vulkan)
├── raeen-audio/     Tempest 3D audio emulation
├── raeen-io/        I/O complex & SSD decompression
├── raeen-input/     DualSense & gamepad input
├── raeen-hle/       High-level emulation of PS5 system libraries
└── raeen-gui/       Desktop application (egui)
```

## Status

🚧 **Early Development** — This project is in its foundational stage. Current focus is on binary loading, kernel syscall translation, and basic GPU command decoding.

## Legal

### Reverse Engineering & Copyright

This project is developed through **clean-room reverse engineering** for the purpose of interoperability, consistent with established legal precedent (e.g., *Sony Computer Entertainment, Inc. v. Connectix Corp.*, 203 F.3d 596 (9th Cir. 2000)). Raeen does not contain, link against, or distribute any proprietary Sony code, firmware, SDK materials, encryption keys, or copyrighted game content. Users must legally obtain their own PS5 firmware and game dumps from hardware they own.

Raeen is a research and preservation project. It is not a tool for piracy, and the maintainers do not condone or support the acquisition or distribution of copyrighted games, firmware, or system software by any means other than dumping them from your own console.

### Trademark Notice

"PlayStation", "PS5", "DualSense", and related marks and logos are registered trademarks of **Sony Interactive Entertainment LLC** and/or Sony Group Corporation. **Raeen is an independent, community-developed project and is not affiliated with, sponsored by, endorsed by, or in any way officially connected to Sony Interactive Entertainment or Sony Group Corporation.**

Any references to PlayStation hardware, system software, or file formats in this project — including the project name — are made solely for **nominative and descriptive purposes**: to identify the platform this software interoperates with. No PlayStation logos, fonts, symbols (△ ◯ ✕ □), or other brand assets are used in Raeen's branding or user interface.

All other trademarks and game titles referenced are the property of their respective owners.

## License

Licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.
