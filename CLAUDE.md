# Raeen — PS5 Compatibility Layer (Claude Code)

Clean-room **PS5 emulator / compatibility layer** in Rust. Native x86-64 guest
execution (Zen 2); HLE for Orbis OS; AGC/PM4 → Vulkan; no CPU interpreter.

**License:** GPL-2.0-only. No Sony SDK, keys, or firmware in-tree. Reference
clones live only under gitignored `reference/`. Attribute ports in
`THIRD_PARTY_NOTICES.md`.

**North star:** market-competitive PS5 emulator on Windows first (then Linux) —
install → library → launch → logs → settings → actionable crash reports
(shadPS4-class UX). Gated by M0–M5; never claim a milestone without its
acceptance test.

---

## Commands

```bash
# Build / run (repo root)
cargo build -p raeen-gui
cargo build --release -p raeen-gui
cargo run -p raeen-gui                    # Shell UI → target/debug/raeen.exe

# Tests (prefer scoped when iterating; workspace before claiming done)
cargo test -p raeen-runtime
cargo test -p raeen-firmware
cargo test -p raeen-hle
cargo test -p kyty-core
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check

# Diagnostics (via raeen.exe)
cargo run -p raeen-gui -- --firmware-info path/to/PS5UPDATE.PUP
cargo run -p raeen-gui -- --load-sprx path/to/module.sprx
```

Shell UI verification: follow `.claude/skills/verify` (PostMessage drive +
GetClientRect / screenshots). Never SendKeys/AppActivate.

Config: `config.toml` at repo root (auto-created; don't commit local tweaks).
Games: `Games/<name>/eboot.bin`. Themes: `themes/`.

---

## Architecture

```
Shell (raeen-gui)
  → FirmwareLauncher → raeen-firmware::load_module
      (SELF passthrough/decrypt → .sprx → PT_SCE_DYNLIBDATA → NID link)
  → raeen-runtime::execute_linked | execute_process
      (GuestArena identity map, VEH HLE trampolines, guest stack, FSGSBASE TLS)
  → raeen-hle (libc/libkernel/… NIDs)
  → raeen-gpu / audio / input   (scaffolds → Kyty-backed)
```

| Crate | Role | Maturity |
|-------|------|----------|
| `raeen-core` | Config, logging, errors, constants | MVP |
| `raeen-loader` | ELF/SELF/PKG parse | Partial MVP |
| `raeen-firmware` | PUP/SLB2, SELF, .sprx, NID link (LM0+LM1) | MVP |
| `raeen-runtime` | GuestArena, VEH, stack, TLS, execute_* (Windows) | MVP |
| `raeen-hle` | System lib NIDs (real memory path; stubs elsewhere) | Partial |
| `raeen-kernel` | VMM/VFS/equeue — **on the launch hot path** via raeen-hle; `syscalls/` dispatch still unwired | Partial |
| `raeen-gui` | egui PS5-style Shell + launcher | MVP |
| `raeen-gpu` | AGC/PM4/shader/Vulkan placeholders | Scaffold |
| `raeen-audio` / `input` | Tempest / DualSense | Scaffold |
| `kyty-core` | Kyty `lib/Core` (+ Sys as `sys_*.rs`) Rust port — **orphan**: zero consumers today (the kyty-graphics dep edge was false and was removed 2026-08-03); wire it or its build cost is waste | Phase 1 |

**Hot path rule:** do not abandon `raeen-firmware` / `raeen-runtime` / Shell.
Port into `kyty-*`, then **wire into** `raeen-*`. Orphan ports that never link
are waste.

**Runtime model:** identity-mapped `GuestArena` + VEH trap-and-emulate HLE.
Keep unless a Kyty approach wins with tests. Vulkan 1.3 first; Metal later.
HLE for system libs; LLE only for user-supplied decryptable firmware modules.
Unresolved NIDs: log name+NID loudly — never silent jump-to-null.

---

## Current reality (do not regress)

**Proven:**
- SELF → .sprx → dynlib → NID link (`raeen-firmware`)
- Windows native execute + VEH HLE; guest heap malloc/memset readback `0xAB`
- Dedicated guest stack + minimal FSGSBASE TLS (`fs:[0]`)
- `execute_process` argc/argv/envp/auxv + exit (tested; Shell launches through
  `execute_process_shared_with_control` — wired, see `raeen-gui/src/launcher.rs`)
- Shell Games/ scan + FirmwareLauncher
- `kyty-core` foundation (not on hot path yet)

**M1 walls — ALL CLOSED** (historical record: `docs/homebrew-gap-analysis.md`;
M1 gate green, list kept so nothing here regresses):
1. Shell → `execute_process` / crt0 stack (not bare `execute_linked`) — closed
2. TLS relocs `DTPMOD64`/`DTPOFF64`/`TPOFF64` + `PT_TLS` + `__stack_chk_guard` @ `fs:0x28` — closed
3. Real `printf`/`puts`/`write` (guest strings → host log) — closed
4. `sceKernelLoadStartModule` + NEEDED `.prx` chain — closed
5. Real `scePthreadCreate` (second guest context) — closed

**Permanent wall:** encrypted retail SELF without user keys. Never add Sony
keys, SDK dumps, or proprietary blobs to the repo.

---

## Milestones (hard gates)

| Gate | Done when |
|------|-----------|
| **M0** | ELF/SELF/PKG + .sprx parse; unknown SCE segments logged; CLI inspect green |
| **M1** | Compiler-produced homebrew (not hand-asm): TLS links, crt0 stack via Shell, observable printf/write, in-tree acceptance test |
| **M2** | Real Vulkan draw via **AGC**/PM4; SPIR-V path runs; visible triangle (screenshotable) — **CLOSED** (fixture DCB + `kyty-graphics` SPIR-V + PPM; swapchain = M3) |
| **M3** | Interactive 2D homebrew; pad → `libScePad`; VideoOut flip; audio stub must not hang — **CLOSED** (real synthesized guest reads `scePadReadState` → CPU-draws a framebuffer by input → `sceVideoOutSubmitFlip`; present-from-guest-memory makes CPU-2D visible; acceptance `crates/raeen-runtime/tests/m3_interactive_2d.rs`; pad/audio HLE tested. Swapchain proper = later.) |
| **M4** | One legal commercial 2D title to interactive menu; crash/log shows syscalls/NIDs/GPU faults; save-data host map — **CLOSED** (Minecraft Bedrock, user-owned retail: interactive menu with proven pad focus, `/savedata0/` host map with persisted world loading, logs actionable by outcome — two blockers found and fixed from log lines alone (`eef31c1`, `d21e727`); record `docs/m4-acceptance-minecraft.md`. Not a perf/stability claim.) |
| **M5** | One 3D title produces recognizable frames (glitches OK); shader MVP for that title — **CLOSED** (same recorded Minecraft run: in-world textured 3D terrain + HUD at 56 FPS; shader MVP = Gen5 analysis→SPIR-V→Vulkan path, decisive fix `d21e727` "ps: direct sgprs"; in-tree retail-free tests in `kyty-graphics` (477 green at record time incl. direct-SGPR + full-chain SPIR-V; the suite has since grown past 730 — `cargo test -p kyty-graphics` is authoritative); record `docs/m5-acceptance-minecraft.md`. Recognizable ≠ correct: MRT1–7/fast-clear/DCC skipped; MVP scoped to this title.) |

Never start M2 graphics while M1 crt0/TLS/module-load still blocks real
binaries — unless the user explicitly prioritizes a graphics spike.

**M2 is AGC, not GNM** (measured 2026-07-15 against the retail title). GNM is
the PS4 API; PS5 titles call **AGC**. The measured title imports 48 `libSceAgc`
+ 7 `libSceVideoOut` + 5 `libSceAgcDriver` functions and **zero** `libSceGnm*`.
This is a naming fix, **not** a re-architecture, and it does **not** cancel the
Kyty port:

* PM4 is the GPU **command stream**; GNM and AGC are two API layers that both
  emit it. Kyty's AGC path emits *more* PM4 than its GNM path, and both converge
  on one command processor (`GraphicsRunSubmit`).
* Kyty has **both**: `Gen4` (GNM) 681 lines, `Gen5` (AGC) **1407** lines. Only
  681 of Graphics' 38,506 lines (1.8%) is GNM-specific. The mass —
  `ShaderSpirv` 8103, `GraphicsRender` 5676, `GraphicsRun` 4269 (the PM4
  processor), `ShaderParse` 3424 — is generation-agnostic. **Port Gen5 + the
  shared core; skip Gen4** (zero imports in this title).
* `crates/raeen-hle/src/libsce_agc.rs` **already exists** (2431 lines, SharpEmu
  port) and matches 25 of the title's 48 AGC NIDs. Kyty Gen5 matches 26. Union
  = 37/48 registered — but that counts *registered symbols, not working ones*
  (Kyty Gen5 carries 119 `EXIT_NOT_IMPLEMENTED`). Coverage ≠ rendering.
* **Unmeasured and decisive for M2 scope:** how much AGC is header-inlined into
  the title. 48 imported entry points against 718,983 relocations suggests much
  of AGC may be inlined, in which case the title emits PM4 directly and the PM4
  interpreter — not the 48 HLE entry points — is the whole M2 deliverable.
  Measure this before scoping M2.

---

## Kyty port (useful code only)

Plan: `docs/superpowers/plans/2026-07-13-kyty-full-port.md`  
Source: `reference/kyty` (gitignored, MIT © InoriRus)  
Ledger: `.superpowers/sdd/progress.md`

**Useful = on the path to M1–M5.** Skip/defer: demos, unused Scripts/lua,
gtest, vendored third-party C (use workspace crates: `ash`, `zstd`, …).

| Phase | Crate | Maps to |
|-------|-------|---------|
| 1 Foundation | `kyty-core` (+ Sys submodules) | — |
| 2 Loader | `kyty-loader` | `raeen-loader` / firmware |
| 3 Kernel | `kyty-kernel` | `raeen-kernel` / HLE |
| 4 Libs | `kyty-libs` | `raeen-hle` |
| 5 Graphics | `kyty-graphics` (~38.5k) | `raeen-gpu` — crown jewel / M2+ |
| 6 Top | `kyty-emulator` | runtime + GUI wiring |

**Faithfulness:** same structure/behavior/API; containers = thin `std`
wrappers with Kyty method names; no manufactured unsafe for C++ scaffolding;
FFI/`unsafe` only with `SAFETY:` notes (Sys/Graphics).

Also study **KytyPS5** (`reference/kytyps5` when cloned) — live PS5 fork of
Kyty already booting commercial 2D/3D; patterns for graphics/pthread/SRT/VM,
not a blind merge.

---

## Reference ecosystem (speed-up sources)

Clone under `reference/` only (gitignored). Update `THIRD_PARTY_NOTICES.md`
when incorporating.

### Port / adapt (license OK with GPL-2.0-only)

| Project | License | Use for |
|---------|---------|---------|
| [InoriRus/Kyty](https://github.com/InoriRus/Kyty) | MIT | Primary port source (in progress) |
| [Nmzik/KytyPS5](https://github.com/Nmzik/KytyPS5) | Kyty/MIT lineage | Active PS5 Kyty — graphics, pthread, LibUlt, SRT, VM |
| [shadps4-emu/shadPS4](https://github.com/shadps4-emu/shadPS4) | GPL-2.0 | Best Orbis HLE reference (memory, libkernel, linker, Vulkan) |
| [par274/sharpemu](https://github.com/par274/sharpemu) | GPL-2.0 | Alternate PS5 architecture (eboot, PRX, VideoOut, DualSense) |
| [PS5Dev/PS5SDK](https://github.com/PS5Dev/PS5SDK) | GPL-2.0 | Real ELF fixtures, CRT, dlsym/module init |

### Fixtures / NIDs / formats

| Project | Notes |
|---------|-------|
| [ps5-payload-dev/sdk](https://github.com/ps5-payload-dev/sdk) | Dynamic link SDK, `prospero-nid`, SCE stubs — check license before port |
| [OpenOrbis/LibOrbisPkg](https://github.com/OpenOrbis/LibOrbisPkg) | PKG/SFO/PFS for loader/library |
| OpenOrbis `create-fself` | Fake-SELF test modules |

### Study only (GPL-3.0 — incompatible unless Raeen relicenses)

| Project | Notes |
|---------|-------|
| [OpenOrbis-PS4-Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) | Homebrew/NID knowledge; do not copy code into this tree |
| fpPS4 | Deep PS4 lib RE that fed shadPS4 — understand, don't transliterate |

**Priority:** KytyPS5 → shadPS4 → PS5SDK/payload-dev fixtures → SharpEmu when stuck.
Ignore ShadPS5 marketing site until a real public tree exists.

---

## Session protocol

1. Read `.superpowers/sdd/progress.md` + `docs/homebrew-gap-analysis.md` + relevant spec/plan
2. Pick the **single highest red gate** (usually earliest M#)
3. TDD: failing acceptance test → implement → green
4. `cargo test` (scoped, then dependents / workspace)
5. Shell/UI → verify skill
6. Update progress ledger: `module: complete (commit <sha7>, N/N tests)`
7. Commit only if user asked; end with unlocked / next gate / risks

### Project agents (`.claude/agents/`)

| Agent | When to use |
|-------|-------------|
| `milestone-driver` | Continue work / next gate / M0–M5 critical path |
| `kyty-porter` | Kyty → Rust batches + integration |
| `emulator-reviewer` | Review unsafe/ABI/HLE/clean-room after emulator edits |
| `hle-stubber` | NID implementations, LoadStartModule, printf, pthread |
| `gpu-pipeline` | M2+ PM4/Vulkan/shader/VideoOut (after M1 unless spike) |
| `shell-ui` | egui Shell, launcher wiring, verify loops |

### Project skills (`.claude/skills/`)

| Skill | Purpose |
|-------|---------|
| `m1-homebrew` | M1 wall order A–E + acceptance |
| `kyty-port-batch` | One Kyty batch TDD + ledger |
| `acceptance-gate` | Honest M0–M5 done criteria |
| `clean-room` | License / no Sony blobs |
| `scoped-cargo` | Package-scoped test/clippy map |
| `verify` | Build/drive/screenshot Shell on Windows |

Also use Superpowers when available: `subagent-driven-development`,
`test-driven-development`.

Work on **main** unless user asks for a branch. Prefer irreversible M1 → M2
progress over speculative refactors.

**Claude Code setup:** `.claude/settings.json` allows cargo/git read; PreToolUse
blocks edits under `reference/`, `PS5 Firmware/`, `*.PUP`; PostToolUse rustfmt
on `.rs`. Restart Claude Code once after the first add of `.claude/agents/`.

---

## Key docs

| Doc | Purpose |
|-----|---------|
| `implementation_plan.md` | Phases M0–M5, architecture |
| `docs/homebrew-gap-analysis.md` | Honest M1 walls |
| `docs/superpowers/plans/2026-07-13-kyty-full-port.md` | Kyty port roadmap |
| `docs/superpowers/specs/*` | Runtime, guest AS, stack/TLS, crt0, Shell, LLE spine |
| `docs/superpowers/plans/*` | Executable task plans (LM0/LM1, RT2, …) |
| `THIRD_PARTY_NOTICES.md` | Attribution + license boundaries |
| `docs/reference-port-ledger.md` | Per-module reference port status; delete `reference/<name>` only when fully ported |
| `.superpowers/sdd/progress.md` | Session ledger |

---

## Gotchas

- **Windows-only runtime today** — non-Windows `GuestArena` returns `MapFailed`; keep `#[cfg]` honest
- **raeen-kernel IS on the launch hot path** (stale claim corrected 2026-08-03) —
  raeen-hle routes mmap/VFS/save-data/equeue through `OrbisKernel` (~400 refs);
  only the `syscalls/` dispatch and `process.rs` remain unwired orphans, and
  `syscalls/file.rs::sys_read` returns a zero-filled placeholder buffer — never
  wire syscall intercept without fixing that first
- **Synthetic fixtures ≠ real homebrew** — hand-built modules skip crt0/TLS/NEEDED; don't mark M1 done on them
- **Shell launch wiring** — `execute_process` exists and is tested; FirmwareLauncher must call it for real `_start`
- **egui Home layout** — never nest `bottom_up` for bottom-anchored Home content (overflow bugs); use painter + explicit rects (`shell/home.rs`)
- **Verify skill** — redefine Win32 `Add-Type` wrappers each PowerShell invocation with a fresh class name
- **Kyty Core scaffolding** — most containers are thin std wrappers; real substance is Graphics (Phase 5)
- **Sys in kyty-core** — not a separate crate (Core↔Sys cycle); `sys_*.rs` + `#[cfg(windows)]`
- **config.toml / Games/** — local; don't commit machine-specific paths

---

## First actions (if starting cold)

1. `cargo test --workspace` (or runtime+firmware+hle+gui) — confirm green
2. M1 wall #1: FirmwareLauncher → `execute_process`; `_start` fixture reads argc
3. TLS relocs + `PT_TLS` + stack canary
4. `printf`/`write` observability
5. After real toolchain Hello World: Kyty Graphics → M2 triangle

Stay on the critical path. Optimize for irreversible progress toward M1, then
M2, then market-relevant titles.
