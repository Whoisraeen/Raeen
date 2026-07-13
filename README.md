# XPS5X — PlayStation 5 Emulator

<p align="center">
  <strong>A cross-platform PS5 compatibility layer that translates PS5 system calls and GPU commands to run natively on your PC hardware.</strong>
</p>

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

This project is developed through **clean-room reverse engineering**. It does not contain any proprietary Sony code, firmware, or SDK materials. Users must legally obtain their own PS5 firmware and game files.

## License

Licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.
