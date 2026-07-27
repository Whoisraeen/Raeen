# Raeen Improvement Checklist

Created 2026-07-27 by a Claude Fable 5 session, immediately after M4 was met
(Minecraft in-world, 56 FPS, `docs/m4-acceptance-minecraft.md`). This file is
the working plan for the "suggest improvements → add all of that" request.
**Any continuation session (Fable, Opus, or human) should work from this file
top-down, update statuses in place, and keep it committed.**

## Conventions

- `[ ]` not started · `[~]` in progress (note branch/agent) · `[x]` done
  (note commit sha + test evidence) · `[!]` blocked (note why)
- TDD per session protocol: failing acceptance test → implement → green.
- Scoped `cargo test -p <crate>` while iterating; workspace-ish before claiming
  done. Update `.superpowers/sdd/progress.md` when an item completes.
- Never claim M-gates without the `acceptance-gate` skill criteria.

## Environment gotchas (verified, do not rediscover)

- **AppControl policy (os error 4551)** blocks `cargo clippy` and some freshly
  built test/bench exes locally. Use `cargo check` + scoped tests; CI
  (`.github/workflows/ci.yml`, windows-latest) is the clippy gate of record.
- **Parallel cargo builds on one `target/` deadlock.** Parallel building
  agents must use git worktrees. User may also have parallel Codex sessions
  holding `target\debug\*.exe` locks.
- A running `raeen.exe` keeps OLD code after rebuild — check process StartTime
  vs binary mtime, not the log.
- Primary display is portrait 1080×1920; Shell screenshots of bottom-anchored
  UI must capture y≈1830, not the top 1080 rows.
- `reference/`, `PS5 Firmware/`, `*.PUP` are edit-blocked by hooks. Retail
  game installs are user-owned, local-only, never committed.

---

## P0 — Critical path (engine)

### 1. M5 acceptance evaluation + record (Minecraft)
- [~] IN PROGRESS — delegated to `milestone-driver` agent (worktree), 2026-07-27.
- **What:** Evaluate honestly (acceptance-gate skill) whether M5 — "one 3D
  title produces recognizable frames (glitches OK); shader MVP for that
  title" — is already met by the Minecraft evidence (in-world 3D render, real
  block textures, Gen5 shader path → SPIR-V, 56 FPS). If yes: in-tree
  acceptance test + `docs/m5-acceptance-minecraft.md` mirroring the M4 record
  (reproduce steps, honest limitations). If no: record the precise gap as the
  new top blocker.
- **Why:** Cheapest irreversible win; converts implicit progress into a
  closed gate before expensive GTA V work.
- **Where:** `docs/m4-acceptance-minecraft.md` (template),
  `crates/raeen-runtime/tests/` (M3 test as pattern), CLAUDE.md milestone table.

### 2. GTA V wall: AGC ACB / async compute breadth
- [~] Phase A IN PROGRESS — delegated to `gpu-pipeline` agent (worktree), 2026-07-27.
- **What (phased):**
  - [~] **A. `*GetSize` families + ACB builder scaffolding.** The 83 missing
    `libSceAgc` NIDs (`docs/gta5-blocker-analysis-2026-07-27.md`) are
    dominated by `sceAgcAcb*` and `*GetSize`. GetSize functions are mechanical
    (return command-packet sizes); implement against Kyty Gen5 / KytyPS5
    reference semantics with tests. Scaffold the ACB (async compute buffer)
    builder mirroring the DCB builder structure.
  - [ ] **B. Compute queue in the PM4 processor.** `kyty-graphics` run.rs
    (`GraphicsRunSubmit` path) currently drives the graphics DCB; add an ACB
    submission path + compute dispatch handling. KytyPS5 is the pattern source.
  - [ ] **C. `libSceAmpr` (46 NIDs)** — async memory prefetch; honest
    semantics (likely synchronous completion) after A/B.
  - [ ] **D. Re-measure GTA V** stop point; update
    `docs/gta5-blocker-analysis-2026-07-27.md`.
- **Why:** One structural capability (async compute) unblocks the whole AAA
  class, not just GTA V.
- **Where:** `crates/raeen-hle/src/libsce_agc.rs`, `crates/kyty-graphics/`
  (run.rs, gen5), `reference/kyty` + `reference/kytyps5` (study/port, MIT).

### 3. Tier-B offline stubs for online/social/service libs
- [~] IN PROGRESS — delegated to `hle-stubber` agent (worktree), 2026-07-27.
- **What:** Honest *offline* semantics (policy already in ledger 2026-07-25:
  no blanket zero-stubs that poison the blocker signal). Targets by measured
  demand: `libSceNpWebApi2` (17), `libSceVoice` (15), `libSceNpManager` (10),
  `libSceAvPlayer` (10), `libSceHttp2`, common dialogs. Semantics:
  not-signed-in / offline errors, empty result sets, async ops that complete
  with error — never fake success. Each function logged first-call.
- **Why:** ~188 of GTA V's missing NIDs and the cross-game gap are this
  surface; single-player boots shouldn't die on online plumbing.
- **Where:** `crates/raeen-hle/src/` (new/existing per-lib modules), shadPS4
  as reference for error codes/struct layouts.

### 4. UE5 diagnosed-but-open fixes
- [~] IN PROGRESS — delegated to `hle-stubber` agent #2 (worktree), 2026-07-27.
- [~] `hle_stack_chk_fail` must NOT return 0 (Until Dawn walks into UD2).
  Terminate the guest with an actionable report instead.
- [~] getdents/fstat layout audit vs shadPS4 — Until Dawn's
  `/app0/deepfiles` empty dir returned 0x200 twice (overflow smell) before the
  canary trip.
- [ ] Dragon Ball: worker threads deref a count (rax=2 → 0x20) as a list
  pointer at module+0x241c820 after WaitEqueue timeouts — needs fault-region
  disassembly follow-through (started, see ledger 2026-07-25/26 ITEM 2).
- **Where:** `crates/raeen-hle/` (stack_chk, getdents/fstat),
  `crates/raeen-kernel/` VFS, ledger ITEM 2 notes.

### 5. Ordered GPU side effects — steps 3–5
- [ ] Step 3: unify the two timestamp clocks (A/B carefully — ASTRO
  timestamp-fence regression territory, per ledger).
- [ ] Step 4: events/EOP ordering under `RAEEN_DEFER_GPU_SIDE_EFFECTS`.
- [ ] Step 5: flip-pending ordering.
- **Context:** Steps 0–2 landed (gate + fail-open + IT_DMA_DATA in-stream,
  `cp_op_it_dma_data` in kyty-graphics run.rs). Design notes in ledger
  2026-07-25/26 ITEM 4.

### 6. Real Vulkan swapchain + GPU-resident present phase 2
- [ ] Phase 2: cross-process image sharing (`VK_KHR_external_memory_win32` →
  wgpu-hal import on the Shell side) — kills the last CPU copy.
- [ ] Swapchain-proper design (ITEM 3 explore pass was in flight — find/redo
  notes; SharpEmu MAILBOX model; MC default-OFF policy).
- [ ] Phase 3: `RAEEN_HOST_GPU_FRAMES` + GPU plugin passes (ABI v2 already
  shipped and forward-compatible).
- [ ] Phase 4 (moat, gated on M5 titles): PM4 motion-vector/depth extraction.
- **Where:** `docs/superpowers/plans/2026-07-27-gpu-resident-present.md`
  (the phased plan), `crates/raeen-gpu/` present path + `present_plugin`.

### 7. Synchronous guest callbacks (call back INTO guest from HLE)
- [ ] Design + implement re-entrant guest dispatch from inside a VEH/gateway
  HLE handler (nested execute on the current guest thread; RSP discipline;
  re-entrancy tests).
- [ ] Retire the blocked class: `qsort` (comparator), `atexit` chain, module
  init/fini callbacks; later VideoOut/GPU event callbacks.
- **Where:** `crates/raeen-runtime/` (dispatch/VEH), then
  `crates/raeen-hle/src/libc.rs` consumers.

### 8. AIO infrastructure
- [ ] The 5 skipped kernel AIO NIDs ("no infra — do not fake", ledger
  2026-07-25). Implement a real host-threadpool-backed AIO with Orbis
  semantics (submit/poll/wait/cancel) in `raeen-kernel`, then the HLE surface.

---

## P1 — Robustness / tooling

### 9. Soak test
- [ ] `cargo xtask soak` (or test-ignored harness): drive Minecraft ≥30 min
  with synthetic pad input; assert no frozen-frame window (frame-epoch must
  advance), capture CPU/core usage.
- [ ] Re-verify the in-world hang that did NOT reproduce post-fix (mutex
  `0x1019a1d48c0` / `0x1019a1d32e0`); if it recurs, instrument holder call
  path via `RAEEN_TRACE_HLE` + owner name.
- **Note:** needs the user's local game install; agents can build the harness
  but a human-supervised run closes it.

### 10. Promote the baseline runner to `cargo xtask baseline`
- [ ] Port `scratch/run-baseline-parts.py` (chunked per-game boot driver,
  merges into `latest.json`) into xtask; add `baseline diff` between runs.
- **Why:** regression tripwire while AGC/stub work churns; feeds item 11.

### 11. Compatibility badges in the Shell library
- [ ] Per-title status chip (Boots / Menu / In-game / Playable) derived from
  local run history (baseline output + last-session outcome), shown on tiles
  and in per-game overlay.
- **Where:** `crates/raeen-gui/src/shell/` (tiles, per-game overlay),
  storage next to per-game config.

### 12. Symbolized / actionable crash report view
- [ ] Pair `logs/crashes/*.dmp` (minidumper already wired) with guest context
  we already have: faulting module+offset, last N HLE/NID calls, unresolved-NID
  inventory, GPU state summary → one shareable report file + Shell view.
- **Where:** `crates/raeen-gui` (view), runtime fault plumbing
  (d87765a crash reporting groundwork).

### 13. Clippy debt: kyty-graphics recompile.rs (~91 clippy-1.97 lints)
- [ ] Fix; verify via CI (local clippy is AppControl-blocked). Keep `-D
  warnings` green.

---

## P2 — Performance

### 14. Async pipeline compilation
- [ ] Compile new Vulkan pipelines off the render thread; skip-draw (or
  fallback) until ready; persistent shader cache already exists — this kills
  first-encounter hitching. Measure with the perf HUD (item 15).
- **Where:** `crates/kyty-graphics` pipeline creation path, `raeen-gpu`.

### 15. In-Shell perf HUD
- [~] IN PROGRESS — delegated to `shell-ui` agent (worktree), 2026-07-27
  (bundled with item 18).
- **What:** toggleable overlay: FPS, frame time (avg/p99), guest CPU core
  usage, upload/present µs (values already measured ad-hoc via puffin scopes
  `execute_dcb_cp` / `publish_frame` / `shell_update` and `egui_upload_us`).
- **Where:** `crates/raeen-gui` overlay + `raeen-gpu` counters; puffin feature
  already wired (`RAEEN_PROFILE=1`).

### 16. MRT1–7 + fast-clear (DCC/CMASK/FMASK/CLEAR_WORD) register support
- [ ] The "143 distinct unknown context registers" triage (ledger 2026-07-27
  late) — implement MRT1–7 colour-buffer blocks + fast-clear; keep skipping
  compression metadata deliberately where safe. Schedule with item 2 (same
  register-decode neighborhood).

---

## P3 — Features / differentiators

### 17. Local trophy store
- [ ] Back `sceNpTrophy2*` with a real local unlock store (per-title, per-user
  JSON/SQLite under savedata-like host map) + Shell trophies page. Serves
  Tier-B necessity AND user delight.

### 18. Screenshot hotkey
- [~] IN PROGRESS — delegated to `shell-ui` agent (worktree), 2026-07-27
  (bundled with item 15).
- **What:** hotkey + pad chord dumps the current presented frame (Shell
  already owns the published frame buffer) to `screenshots/` with title id +
  timestamp; toast on success (egui-notify already wired).

### 19. DualSense passthrough
- [ ] Rumble first (gilrs ff or hidapi), then haptics/adaptive triggers via
  DualSense output reports. Un-reserve the Settings toggle ("DualSense
  Features" currently labeled reserved — see 2026-07-27 settings-backend
  audit).

### 20. Auto-updater
- [ ] Close the loop on the existing "update staged" toast: download →
  hash-verify → swap on next restart. Windows-first.

### 21. Video clip capture (stretch, after 18)
- [ ] Rolling ring of recent frames → mp4 on hotkey. Only after screenshot +
  perf work; consider `re_mp4`/ffmpeg-sidecar licensing (GPL-2.0-only tree).

---

## Session log (append per session)

- **2026-07-27 (Fable 5):** Checklist created. Delegated in parallel
  (worktree-isolated agents): item 1 → milestone-driver, item 2A →
  gpu-pipeline, item 3 → hle-stubber, item 4 → hle-stubber, items 15+18 →
  shell-ui. Integration + status updates to follow as agents land.
