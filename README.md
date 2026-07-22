# XPS5X — PlayStation 5 Emulator

<p align="center">
  <strong>A cross-platform PS5 compatibility layer that translates PS5 system calls and GPU commands to run natively on your PC hardware.</strong>
</p>

<p align="center">
  <a href="https://github.com/Whoisraeen/XPS5X/actions/workflows/ci.yml"><img src="https://github.com/Whoisraeen/XPS5X/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Whoisraeen/XPS5X/releases/latest"><img src="https://img.shields.io/github/v/release/Whoisraeen/XPS5X?include_prereleases&label=download" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--2.0--only-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://whoisraeen.github.io/xps5x-website/"><strong>Website</strong></a> ·
  <a href="https://github.com/Whoisraeen/XPS5X/releases/latest">Download (Windows x64)</a> ·
  <a href="https://github.com/Whoisraeen/XPS5X/issues">Issues</a>
</p>

> **Status: early alpha.** XPS5X currently runs homebrew-style test binaries
> (process launch, TLS, printf/write observability). It does not play
> commercial games yet. Releases include a built-in auto-updater
> (Settings → System) that downloads new versions from GitHub Releases and
> applies them on restart.

---

## What is XPS5X?

XPS5X is a PS5 emulator / compatibility layer that enables PS5 game binaries to run on Windows, Linux, and macOS. Since the PS5 uses an x86-64 CPU (AMD Zen 2), XPS5X **natively executes** game code on your PC — no slow interpretation needed. The magic happens in the **translation layers**:

- **Kernel HLE** — Translates PS5's Orbis OS (FreeBSD-based) system calls to your host OS
- **GPU Translation** — Converts Sony's GNM/PM4 GPU commands to Vulkan (or Metal on macOS)
- **Shader Recompiler** — Recompiles RDNA2 ISA shader binaries to SPIR-V in real-time
- **Hardware Emulation** — Emulates Tempest 3D Audio, the custom I/O complex, and DualSense features

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    XPS5X Application                     │
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
git clone https://github.com/XPS5X/xps5x.git
cd xps5x

# Build in release mode
cargo build --release

# Run
cargo run --release
```

## Project Structure

```
crates/
├── xps5x-core/      Core engine, configuration, error handling
├── xps5x-loader/    SELF/ELF/PKG binary parser and loader
├── xps5x-kernel/    Orbis OS kernel HLE (syscall translation)
├── xps5x-gpu/       GPU command translation (GNM→Vulkan)
├── xps5x-audio/     Tempest 3D audio emulation
├── xps5x-io/        I/O complex & SSD decompression
├── xps5x-input/     DualSense & gamepad input
├── xps5x-hle/       High-level emulation of PS5 system libraries
└── xps5x-gui/       Desktop application (egui)
```

## Status

🚧 **Early Development** — This project is in its foundational stage. Current focus is on binary loading, kernel syscall translation, and basic GPU command decoding.

## Legal

### Reverse Engineering & Copyright

This project is developed through **clean-room reverse engineering** for the purpose of interoperability, consistent with established legal precedent (e.g., *Sony Computer Entertainment, Inc. v. Connectix Corp.*, 203 F.3d 596 (9th Cir. 2000)). XPS5X does not contain, link against, or distribute any proprietary Sony code, firmware, SDK materials, encryption keys, or copyrighted game content. Users must legally obtain their own PS5 firmware and game dumps from hardware they own.

XPS5X is a research and preservation project. It is not a tool for piracy, and the maintainers do not condone or support the acquisition or distribution of copyrighted games, firmware, or system software by any means other than dumping them from your own console.

### Trademark Notice

"PlayStation", "PS5", "DualSense", and related marks and logos are registered trademarks of **Sony Interactive Entertainment LLC** and/or Sony Group Corporation. **XPS5X is an independent, community-developed project and is not affiliated with, sponsored by, endorsed by, or in any way officially connected to Sony Interactive Entertainment or Sony Group Corporation.**

Any references to PlayStation hardware, system software, or file formats in this project — including the project name — are made solely for **nominative and descriptive purposes**: to identify the platform this software interoperates with. No PlayStation logos, fonts, symbols (△ ◯ ✕ □), or other brand assets are used in XPS5X's branding or user interface.

All other trademarks and game titles referenced are the property of their respective owners.

## License

Licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.
