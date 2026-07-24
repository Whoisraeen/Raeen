# Raeen Strategy Report — Becoming the Go-To PS5 Emulator

**Date:** 2026-07-23
**Scope:** Competitive position vs the two closest references, the feature roadmap to
"go-to" status, the PS1–PS5 multi-generation scope decision, and standout
differentiating features (incl. an honest verdict on DLSS frame generation).

**Method note.** Every claim below was produced by reading the three actual
codebases (Raeen + `reference/sharpemu` + `reference/kytyps5`), with each
"Raeen is better" claim adversarially re-checked against what the references
actually do, and with the frame-gen license facts confirmed against current
upstream terms. Findings that didn't survive that check were dropped or
softened. This report is deliberately unflattering where the evidence demands
it — consistent with the project's `acceptance-gate` honesty discipline.

**One-line status (do not let any section obscure it):** Raeen boots **zero
commercial titles** today (M2/M3 on synthetic fixtures). It has an unusually
strong shell and architecture for its maturity, but every path to an unmodified
retail title still ends in a stub. Nothing in the "differentiation" sections
matters until the P0 walls fall.

---

## 0. Executive summary

Three decisions, three sentences:

1. **Position as "the best-designed, most PS5-native emulator," not "runs
   everything."** Raeen's genuine edges are engineering / UX / process, and —
   critically — **there is no entrenched PS5 incumbent yet** (shadPS4 is a mature
   *PS4* emulator whose PS5 back-compat is an unstarted 2027 roadmap item). The
   PS5 frontier is open; that is Raeen's real opportunity.
2. **Stay x86-64: PS4 + PS5 only. Never build an in-house MIPS/PPC/Cell engine.**
   Raeen's native-execution model runs *only* x86-64 guests. PS4 is the one
   aligned adjacent (same ISA, GNM→PM4 reuse, PS5 runs PS4 titles) — but it must
   wait behind the first real PS5 boot.
3. **On enhancement features: right instinct, wrong vendor.** Frame gen /
   upscaling is the biggest visual lever an emulator has, and Raeen is
   structurally positioned to do it *better than anyone* by feeding real motion
   vectors from the PM4 stream — but with **FSR (MIT), not DLSS (proprietary,
   GPL-incompatible)**.

The through-line: **close the P0 walls and boot one real game first.** The
differentiators are the destination, not the next commit.

---

## 1. Competitive position — Raeen vs SharpEmu vs KytyPS5

On the only test that ultimately matters — running games — Raeen is *behind*
both references: **KytyPS5** (C++) boots commercial 2D and 3D titles to
gameplay; **SharpEmu** (C#) runs real titles out-of-process with a hosted Vulkan
surface. So "better" here means engineering / architecture / process — and most
of Raeen's genuine edges hold against only *one* of the two references.

### 1.1 Where Raeen genuinely leads

**Strong (holds against both):**

- **The only true PS5-console 10-foot shell of the three.** `crates/raeen-gui/src/shell/`
  (boot, home, control-center, pad/keyboard nav, PS-hold-to-quit). SharpEmu is an
  Avalonia desktop window; KytyPS5 is a Qt dialog that spawns the emulator in an
  external `cmd.exe`. Neither attempts a console UX.
- **A documented adversarial self-audit ledger**, not a README disclaimer.
  `.superpowers/sdd/progress.md` records *falsified* hypotheses;
  `docs/homebrew-gap-analysis.md` enumerates honest walls. Both references confine
  honesty to a one-time README snapshot.

**Moderate (mostly edges over *one* reference):**

- **Widest on-disk container parsing** — Raeen alone decodes the binary PKG
  `\x7FCNT` header, `\0PSF` PARAM.SFO, and SLB2/PUP firmware
  (`crates/raeen-loader/.../pkg.rs`, `slb2.rs`, `pup.rs`); both refs read only
  `param.json`. *Caveat: PKG path is metadata-only; the param.json reader is a
  fragile line-matcher.*
- **Prioritized unresolved-import diagnostics** — `raeen-gui/src/main.rs:315`
  groups unknown NIDs by library, ranked by reloc count → an "implement this
  library next" list. KytyPS5 dumps flat; SharpEmu throws.
- **Deterministic, provider-retaining NID handling + spelling normalization**
  (`nid.rs`) — stable winner, no lost bindings, strips `.native/.sprx/.prx`
  (measured on Minecraft PPSA17221). *(Edge over SharpEmu; KytyPS5 already keys
  more granularly.)*
- **Bidirectional NID codec** — Raeen can *decode* an 11-char SCE-base64 NID;
  both refs are encode-only.
- **Never-silent skip-and-continue on unknown PM4 ops** (`kyty-graphics/src/run.rs:20`)
  vs KytyPS5's hard `EXIT`. *(Edge over KytyPS5 only; partly a crutch for Raeen's
  incompleteness.)*
- **Statement-level `unsafe` with per-site `// SAFETY:` notes** (~463/372) — finer
  audit granularity than SharpEmu's whole-class `unsafe` or KytyPS5's raw
  `reinterpret_cast`. *~80% honored; aspirational, not enforced.*
- **GPLv2 link-compatibility auditing of deps** (`THIRD_PARTY_NOTICES.md`) — pins
  rspirv/naga to their MIT option; KytyPS5 vendors Apache-2.0 SPIRV-Tools into a
  GPL-2.0 tree with no comment.
- **Real `libSceAudioOut2` PCM playback** where SharpEmu only paces silence.
  *(Tie with KytyPS5 — Raeen ported the model from it.)*
- **Per-title play/fault ledger in-launcher**, **gate-named acceptance tests in
  CI**, and a **self-contained single `cargo build`** (KytyPS5 needs
  CMake+Ninja+Qt6+clang-cl+Vulkan SDK + 13 submodules). *(vs KytyPS5; SharpEmu's
  `dotnet build` is equally one-tool.)*

### 1.2 Where Raeen is still behind (candid)

- **Games actually running** — KytyPS5 boots real 2D+3D; Raeen's "CLOSED" M2/M3
  gates are on synthetic fixtures that fail Raeen's *own* acceptance standard.
- **HLE trap mechanism — the biggest structural gap.** `trampoline.rs` makes the
  HLE region `PAGE_NOACCESS`, so **every** guest library call faults → VEH →
  dispatch. Both refs use zero-fault thunks. A permanent perf tax at retail call
  volumes.
- **HLE breadth is the smallest** (~889 registered fns vs SharpEmu ~1,070 vs
  KytyPS5 ~1,586); SharpEmu also has a build-fail NID-integrity gate Raeen lacks.
- **GPU/shader maturity** — Raeen has a per-instruction SPIR-V emitter (no
  SSA/CFG/exec-mask), CPU deswizzle, and is **offscreen-only (no swapchain)**.
  KytyPS5 ships a shadPS4-class recompiler with host-GPU detiling.
- **Audio ported not original + codec-blind** (Ajm stubbed). **Controller output
  is input-only** (rumble/triggers/haptics are no-op acks).
- **Crash isolation** — Raeen runs the guest in-process under VEH (a hard fault
  can take down the Shell); both refs isolate out-of-process.
- **Cross-platform** — SharpEmu ships Win+Linux+macOS; Raeen is Windows-only.
  **Machine-checkable license hygiene** — SharpEmu is REUSE-compliant; Raeen's
  edge is reasoning-in-prose.

**Bottom line:** cleanest architecture, widest container parsing, most honest
self-audit, and the only real console shell — sitting on a smaller,
far-less-exercised codebase. Raeen's wins are real but nearly all are
engineering/process/hygiene edges.

---

## 2. The go-to feature roadmap

Go-to status = a stack of trust-and-friction reducers layered on **adequate
real-game compatibility**. Tiered by priority.

### P0 — Table stakes (not a usable emulator without these)

| Feature | Raeen state | Effort |
|---|---|---|
| **Real playable compatibility on wanted titles** ⚠️ *the whole game* | boots only synthetic M2/M3 fixtures — **0 commercial titles** | XL |
| **PSSL→SPIR-V recompiler w/ EXEC-mask + structured CFG** ⚠️ WALL | `crates/raeen-gpu/src/shader/*` emits straight-line SPIR-V only — no exec-mask lowering, no branch reconvergence | XL |
| **AGC/PM4 command-processor completeness (Gen5)** ⚠️ WALL | `agc_exec.rs` + `draw_translate.rs` substantial but many `EXIT_NOT_IMPLEMENTED`; the PM4 interpreter *is* the deliverable | XL |
| **Robust plaintext SELF / fake-SELF loading (NOT decryption)** ⚠️ PERMANENT WALL | `NoKeysProvider` is correct — never chase keys. Ingest OpenOrbis fake-SELF + user-decrypted `eboot.bin`, all `PT_SCE_*` segments | M |
| **`sceKernelLoadStartModule` + NEEDED PRX chain** | registry holds multiple modules; nothing drives chain discovery | L |
| **Multi-threaded guest maturity (per-thread TCB)** | real host threads + pthread HLE exist; correctness under contention unproven | L |
| **Per-game compatibility database (playable/ingame/menu/nothing)** | the #1 artifact players check. `status` field exists, no tier concept. **Highest-leverage win buildable now** — collects data the moment game #1 boots | M |
| **Controller out-of-box (input + output)** | input works; output (rumble/triggers/haptics) is 1-line stubs | M |

### P1 — Competitive (to rival shadPS4)

Real `VK_KHR_swapchain` present ⚠️ (offscreen-only today) · **zero-fault HLE
thunk** ⚠️ (every import faults through VEH) · actionable crash bundle · **Linux
then macOS** (emulation's center of gravity is Steam Deck/handheld) · **out-of-
process crash isolation** (in-process is a *liability*) · DualSense HID output ·
Ajm audio decode (ATRAC9/AAC — stubs can *hang* boot state machines) · VideoOut
flip/vblank timing · resolution upscaling · async shader pre-caching ·
**Discord/community hub** · input remapping.

### P2 — Differentiators (several already ✅)

Config-per-game ✅ · auto-updater ✅ · **PS5-console shell ✅** · save-data mapping
✅ · Rust/clean-arch ✅ · honest self-audit credibility ✅ · trophies · mods ·
**save states ⚠️ architecturally out of reach** (native-exec + VEH-HLE can't
cheaply snapshot host-thread+GPU+HLE state — acknowledge, don't promise) ·
netplay.

---

## 3. Scope decision — should Raeen add PS1–PS5 support?

### Recommendation: **PS4 + PS5 only. Add PS4 — but strictly *after* the first real PS5 boot. Never build an in-house MIPS/PPC/Cell engine.**

**The ISA is the strategy.** Raeen has no interpreter/dynarec/JIT by design —
`crates/raeen-runtime` executes the guest's `e_entry` *directly as a native
sysv64 function* on the host x86-64 core. That one fact partitions the
generations cleanly and unforgivingly:

| Gen | CPU | Runs native on Raeen? | Mature free incumbent |
|---|---|---|---|
| PS1 | MIPS R3000A | ❌ needs a MIPS interpreter/dynarec | DuckStation (≈solved) |
| PS2 | Emotion Engine MIPS + VU0/VU1 | ❌ needs microVU-class recompilers | PCSX2 (20+ yrs) |
| PS3 | Cell PPE + SPE (PowerPC) | ❌ needs PPU **and** SPU recompilers + RSX translator | RPCS3 |
| **PS4** | **x86-64 Jaguar + GCN** | ✅ **identical GuestArena + VEH path** | shadPS4 (overlaps) |
| **PS5** | **x86-64 Zen 2 + RDNA2** | ✅ **the core target** | *none yet* |

"Add PS1/2/3" is not scope expansion — it is **founding a second, unrelated
emulator that shares nothing with Raeen but the menu bar**, aimed at three
unbeaten incumbents, and it discards the one design decision that makes Raeen
coherent. The "universal PlayStation" brand is real but collapses on inspection:
you'd ship three worst-in-class foreign-ISA cores under a shared launcher. If
"universal" is ever pursued, do it by **bundling the existing open cores
(DuckStation/PCSX2/RPCS3) under one shell** — a distribution decision for a
far-future Raeen, not an engine one.

**Why PS4 is the one real "yes":** (1) CPU-native — Jaguar is x86-64, runs the
identical path; (2) PM4 GPU reuse — GCN+GNM emits the *same command stream* the
M2 AGC path already interprets (Kyty even ships a GNM/Gen4 path beside AGC/Gen5);
(3) subset of the goal — the PS5 itself runs PS4 titles; (4) it's likely where
Raeen's *first genuinely playable* commercial titles land (huge, DRM-lighter,
tractable back-catalog).

**Phased answer:**
- **NOW — don't preempt.** Nothing new. Close the M1 walls and boot the first
  real title (crt0 stack, TLS relocs + `PT_TLS` + canary @ `fs:0x28`, real
  printf/write, `LoadStartModule` + NEEDED chain, real `scePthreadCreate`). The
  strategic risk is not "too little scope" — it's "still zero real games booting."
- **LATER — after a real PS5 game boots and PM4→Vulkan is proven.** Add PS4:
  extend HLE to Orbis, grow GNM PM4 opcodes on the *existing* engine. Frame as
  "native PS4/PS5 HLE — one product, two entry points."
- **NEVER — as an in-house engine.** MIPS/microVU/Cell+RSX. Building any burns
  runway that belongs to the open PS5 frontier.

---

## 4. Standout differentiating features

### 4.1 The DLSS frame-gen idea: right instinct, wrong vendor

Frame generation and temporal upscaling are the single biggest visual-quality
lever an emulator can pull, and Raeen is *structurally* positioned to do it
better than anyone in the PS-emulator space. But **DLSS specifically is the wrong
bet**, for two reasons, in order:

- **Catch 1 — it's a license failure, not a technical one.** NVIDIA
  DLSS/Streamline binaries are proprietary (forbid source disclosure, derivative
  works, no-charge redistribution) — the inverse of copyleft. Linking or shipping
  them inside a **GPL-2.0-only** binary makes Raeen non-distributable under its
  own license; there is no "system library" cover for a plugin Raeen ships.
  **FSR (FidelityFX) is MIT** — vendorable in-tree. XeSS fails the same wall
  (`libxess.dll` is a closed binary). So this is a *vendor* problem, not a
  *feature* problem.
- **Catch 2 — frame gen wants motion vectors, and that's the actual moat.** Every
  mass-market post-process tool (Lossless Scaling, AMD AFMF2, NVIDIA Smooth
  Motion) has no depth/MVs, so it ghosts on HUDs and shimmers on pans. RPCS3 and
  shadPS4 ship only spatial **FSR1**. **Raeen owns the PM4→Vulkan translation** —
  `draw_translate.rs`/`agc_exec.rs` already key render targets by `CB_COLOR0_BASE`
  and decode depth via `DB_Z_INFO`, so Raeen can identify the title's own depth +
  velocity buffers and hand an upscaler *real game motion vectors*. That is what
  no external overlay or driver shim can replicate — and DLSS would need the exact
  same MV-extraction work, buying nothing FSR doesn't while adding the license
  hazard and RTX-40/50 hardware lock.

**Recommended path (three tiers):**
1. **Zero-integration stopgap:** ship a clean "present for driver FG" mode (stable
   pacing, no UI baked into the flip surface) and document enabling **NVIDIA
   Smooth Motion** (Vulkan support added 2025) or **AMD AFMF2** in the driver
   panel. Nothing links; zero GPL exposure. Frame it honestly as a recommendation,
   not a Raeen-owned feature.
2. **License-clean baseline:** integrate **FSR 3.1** (MIT; FG decoupled from
   upscaling as of 3.1). Build **FSR Super-Resolution first** — it shares ~90% of
   the MV/depth plumbing with FG and de-risks it. Entry point:
   `AgcGpuSession::set_runtime_config` (`resolution_scale`).
3. **The moonshot differentiator:** MV-assisted FSR fed by PM4-extracted motion
   vectors + depth. The genuine moat — gated on GPU maturity (post-M4 into M5)
   because PM4 carries no semantic tag for the velocity MRT (needs MRT1–7 tracking
   + heuristic velocity/depth identification by format `R16G16_SFLOAT`/sizing/
   usage or per-title profiles; fall back to depth-only FSR).

**On user-supplied DLSS:** an optional, user-provided, `dlopen`'d DLL Raeen never
ships is the *only* non-infringing shape — and even that is legally murky. Do not
build it as a first-class feature. FSR + driver FG cover the same ground cleanly.

### 4.2 Ranked standout features (differentiation × feasibility)

**Tier 1 — Unique moats, buildable on already-closed milestones (M2/M3):**

1. **PS5 RE Studio — PM4-level frame capture + auto-populating compat DB.**
   Capture a guest frame at the PM4 layer Raeen already interprets (per-draw
   packets, disassembled bound shaders, render targets, the exact list of
   skipped ops/registers); the anonymized capture doubles as a structured
   per-title compat record. No PS5 emulator productizes command-stream
   inspection — RenderDoc/PIX are blind to PM4 semantics. The hard part is
   *done*: `CommandProcessor` in `kyty-graphics/src/run.rs` already computes
   `distinct_skips`/`refused_draws` and self-describes each skip. Identity:
   **"the emulator that can explain itself."** *Feasibility: M (format + egui
   inspector + shader-disasm view). GPL-clean; store only Raeen metadata, never
   game assets.*
2. **True-feel DualSense — 1:1 adaptive triggers + haptics + lightbar.** Complete
   the DualSense **output** path (report 0x02/0x31 over USB/BT) and wire
   `libScePad`. The line no competitor can honestly claim: **"the only PS5
   emulator that feels like a PS5."** Raeen already *reads* a real pad (`hid.rs`);
   only output is missing (`adaptive_triggers.rs`/`haptics.rs` are stubs).
   *Feasibility: M (rumble+triggers+lightbar), L (full haptic-audio fidelity).
   Highest emotional-differentiation-per-line-of-code on the board.*
3. **Self-describing crash triage — deterministic diagnostics → auto-filed
   reports.** Turn the sequence-numbered, host-timestamp-free diagnostics stream
   (`diagnostics.rs`) + GPU skip-tracking + unresolved-NID logs into one
   reproducible report the Shell files automatically (optionally with a
   plain-English first-divergence summary). Deterministic ordering enables
   regression diffing that log-scraping can't. Productizes the honesty
   discipline; feeds #1. *Feasibility: M. GPL-clean; strip PII/game bytes; AI
   summaries must not fabricate confidence.*

**Tier 2 — Strong wins, standard tech, gated on the real swapchain (M3+):**

4. **Native HDR passthrough** — present the guest's HDR10 swapchain to an HDR host
   surface instead of tone-mapping to SDR. PS5 output is HDR-native (offscreen
   path already renders `R16G16B16A16`/`B10G11R11`), so Raeen passes HDR through
   cleanly rather than faking it. shadPS4 headlined this in v0.7.0. *M–L; needs
   `VK_EXT_swapchain_colorspace`.*
5. **Present-surface FSR1 + super-sampling / dynamic-res override** — for PS5 the
   win is super-sampling + forcing dynamic-res targets to a fixed high res (not
   the low→4K multiplier of Dolphin/PCSX2). *FSR1: M (MIT). Dynamic-res override:
   L–XL, per-title fragile, M4+.*
6. **Built-in post-process chain (CAS/SMAA/LUT/color-grade)** — trivial once we own
   the present surface; removes a ReShade setup step. Long-tail, not a marquee.
   *S once swapchain exists; CAS is MIT — don't bundle ReShade.*

**Tier 3 — Community engines / go-to-market, gated on real games (M4+):**

7. **Community enhancement framework** — per-game patches, cheats, ultrawide, and
   **hash-keyed HD texture replacement reusing the existing content-hash cache**
   (`raeen-gpu/src/vulkan/cache.rs`), layered on `per_game.rs` + the compat DB.
   *L, M4. Keep packs user-directed/out-of-tree (content/legal). Treat
   60fps/ultrawide as community-authored, never first-party promises.*
8. **Handheld-first console experience (Steam Deck / ROG Ally)** — high positioning
   value vs desktop-window rivals, but a go-to-market angle, not a moat, and
   blocked on the Linux port (VEH-HLE needs a Linux signal-handler equivalent).
   *XL, longest lead time.*
9. **Deterministic replay / TAS tooling** — novel for a PS5 emulator, but
   *execution* is not deterministic (native host threads/timing); needs input
   capture + RNG/timer virtualization + schedule control that don't exist.
   *L–XL, M4+. Don't over-promise "deterministic."*

**Explicitly ruled out:**
- **RTX Remix RT injection** — Remix lifts *fixed-function* DX8/9 scenes; PS5
  titles use programmable AGC/PM4 shaders with no fixed-function geometry.
  Infeasible *and* NVIDIA/RTX-locked.
- **Bundling DLSS/XeSS/Lossless Scaling/AFMF/Smooth Motion in-tree** — all
  proprietary or driver features; cannot ship in a GPL-2.0 tree. User-enabled
  externally only.

### 4.3 Sequencing — don't let shiny features jump the P0 queue

The two highest-differentiation visual features (MV-aware FSR upscaling and frame
gen) **cannot be built until real games render** — they depend on MRT1–7 tracking
and velocity heuristics only meaningful post-M4/M5. Do not start a
graphics-enhancement spike while M1 crt0/TLS/module-load blocks real binaries.

- **Buildable now (M2/M3 closed) — genuine near-term standouts:** PM4 RE Studio
  (#1) + crash triage (#3) (substrate exists in `run.rs`/`diagnostics.rs`);
  DualSense output (#2) (only the output report is missing).
- **Unlock with the M3 swapchain:** HDR passthrough (#4), present-surface FSR1
  (#5), post-process chain (#6).
- **Gated behind real games (M4+):** MV-aware FSR → frame gen (the moonshot),
  dynamic-res override, community texture/patch framework, TAS tooling, handheld
  (also waits on the Linux port).

**Recommended order of attack:** (a) ship the zero-cost "present cleanly for
driver FG" mode + docs the moment a swapchain exists; (b) land **DualSense
output** and the **PM4 Studio / crash-triage** pair as the *now* differentiators;
(c) build **FSR Super-Resolution before Frame Gen**; (d) treat MV-assisted frame
gen as the M5-era crown, and never market it (or "deterministic replay") before
the capability is honest.

---

## 5. The through-line

Raeen's opportunity is real and specific: **the PS5 frontier has no mature
incumbent, and Raeen already owns the two assets that make the marquee
differentiators possible — the PM4→Vulkan translator and a real console shell.**
But those assets are worthless until a commercial game boots. The entire strategy
reduces to one ordering:

> **Close the P0 walls → boot one real PS5 game → then, and only then, spend the
> PM4-ownership and DualSense/diagnostics advantages on features nobody else can
> copy. Stay x86-64 (PS4+PS5). Ship FSR, not DLSS.**

**Anchor files for the feasibility claims above:** `kyty-graphics/src/run.rs`
(CommandProcessor self-instrumentation), `crates/raeen-input/src/hid.rs` +
`adaptive_triggers.rs`/`haptics.rs` (DualSense), `crates/raeen-core/src/diagnostics.rs`
+ `crates/raeen-gui/src/shell/launcher.rs` (triage), `crates/raeen-gpu/src/draw_translate.rs`
/`agc_exec.rs` (`CB_COLOR0_BASE`/`DB_Z_INFO` MV/depth extraction),
`crates/raeen-gpu/src/vulkan/cache.rs` (texture content-hash), `crates/raeen-gpu/src/vulkan/mod.rs`
(M3 present boundary).
