# Kyty → Rust Full Port — Master Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development
> to execute each phase task-by-task. This is a multi-session program; each
> phase has its own detailed task list appended as it begins.

**Goal:** Faithfully port the Kyty PS4/PS5 emulator (~100k LOC of its own C++,
`reference/kyty`, MIT © 2021 InoriRus) into XPS5X as idiomatic-but-faithful
Rust, in dependency order, bottom-up.

**Decision of record (user, 2026-07-13):** *Literal 1:1 transliteration*,
*everything*, *dependency order*. Not a from-scratch re-imagining — a faithful
reproduction of Kyty's modules, files, and logic, translated to compiling Rust.

## Faithfulness conventions

"1:1" is interpreted as **same structure, same behavior, same public API**,
not byte-identical C++ idioms:

1. **std-reimplementation types** (Kyty `Vector`, `String`, `String8`,
   `Hashmap`, `LinkList`, `SimpleArray`, `ArrayWrapper`, `MemoryAlloc`,
   `MSpace`, `RefCounter`, `SafeDelete`, `Singleton`): implement as **thin Rust
   wrappers over `std`** exposing Kyty's method names/semantics. Faithful
   behavior, without shipping a worse `Vec`. Downstream ported code compiles
   against the same API.
2. **Bundled 3rdparty** (Kyty's vendored Vulkan/SDL2/lua/zstd/zlib/sqlite/
   gtest): replaced by the workspace's existing Rust crates — `ash` (Vulkan),
   `zstd`/`lz4_flex`, `rusqlite` (if Database needed), `serde_json` (JsonReader),
   `winit`+SDL-not-needed, `mlua`/`rlua` (Scripts, if needed). Never port the
   vendored C.
3. **Data tables** (e.g. `CharUcd.cpp`, 8.4k lines of Unicode data): port as
   data, or back with a `unicode-*` crate exposing the same queries.
4. **OS abstraction** (`lib/Sys/*`, `VirtualMemory`, `Threads`, `File`,
   `Timer`, `DateTime`): map to `std` + the `windows-sys` crate. Windows-first;
   Linux paths ported as feasible or stubbed behind `#[cfg]`.
5. **C++ raw pointers / manual memory** become the safe Rust equivalent
   (slices, `Box`, `Arc`, `Vec`) where possible; genuinely pointer-based guest
   memory keeps `unsafe` with a `SAFETY:` note, matching the rest of XPS5X.

**Clean-room:** Kyty is community MIT code — porting from it is fine, with
attribution (`THIRD_PARTY_NOTICES.md`). But anything that looks like a verbatim
Sony SDK struct layout or NID table is derived from clean sources (OpenOrbis,
our own analysis), never assumed correct because Kyty has it.

## Target crate layout (mirrors Kyty, integrated into the XPS5X workspace)

| Kyty tree | New crate | Maps toward |
|---|---|---|
| `lib/Core` + `include/Kyty/Core` | `crates/kyty-core` | (foundation) |
| `lib/Sys` + `include/Kyty/Sys` | `crates/kyty-sys` | (foundation) |
| `lib/Math` | `crates/kyty-math` | (foundation) |
| `lib/Scripts` | `crates/kyty-scripts` | (config/scripts) |
| `emulator/src/Loader` | `crates/kyty-loader` | ↔ existing `xps5x-loader` |
| `emulator/src/Kernel` | `crates/kyty-kernel` | ↔ existing `xps5x-kernel` |
| `emulator/src/Libs` | `crates/kyty-libs` | ↔ existing `xps5x-hle` |
| `emulator/src/Graphics` | `crates/kyty-graphics` | ↔ existing `xps5x-gpu` |
| `emulator/src/*` (top) | `crates/kyty-emulator` | ↔ `xps5x-runtime` |

Kyty ports land in `kyty-*` crates first (keeps the 1:1 mapping crisp and
un-entangled); integration into the `xps5x-*` runtime happens per-subsystem
once a `kyty-*` crate is functional. This keeps the existing, tested XPS5X
runtime green throughout.

## Dependency-ordered phases

- **Phase 1 — Foundation.** `kyty-core`, `kyty-sys`, `kyty-math`,
  `kyty-scripts`. ~30k LOC. Within this phase, port the pieces the emulator hot
  path actually uses first (Common/DbgAssert/SafeDelete/Singleton → Sys →
  VirtualMemory/Threads/File → String/Vector/Hashmap → the rest), so later
  phases unblock as early as possible.
- **Phase 2 — Loader.** `kyty-loader` (~2.9k). SELF/PKG/ELF; cross-check
  against XPS5X's existing loader.
- **Phase 3 — Kernel.** `kyty-kernel` (~5.8k). HLE threads/memory/sceKernel.
- **Phase 4 — Libs.** `kyty-libs` (~3.9k). HLE library stubs.
- **Phase 5 — Graphics.** `kyty-graphics` (~38.5k). GNM Pm4 → Vulkan, PSSL
  (`ShaderParse`/`ShaderSpirv`) → SPIR-V, VideoOut, tiling. The crown jewel.
- **Phase 6 — Emulator top + integration.** `kyty-emulator` (Audio, Controller,
  Dialog, Network, Config) + wiring the `kyty-*` crates into the XPS5X runtime.

## Ledger

Per-file/module status tracked in `.superpowers/sdd/progress.md` under a
"Kyty port" heading. Each ported module: `<module>: complete (commit <7>, N/N
tests, review clean)`.

---

## Phase 1 detailed task list — appended when Phase 1 begins (below).
