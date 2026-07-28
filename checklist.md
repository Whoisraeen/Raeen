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
- [x] DONE 2026-07-27 — **M5 MET** on committed evidence (commit `f56132f`,
  merged `40396ab`). `docs/m5-acceptance-minecraft.md` written; ledger updated;
  CLAUDE.md milestone table updated (M4+M5 CLOSED). Evidence: 477 kyty-graphics
  tests green incl. the d21e727 direct-SGPR gate test + retail-free
  GCN→validated-SPIR-V chain. All M0–M5 gates now closed — future "milestones"
  are correctness/breadth (items 2, 16) and stability (item 9).
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
- **What (phased):**
  - [x] **A. DONE 2026-07-27** (commit `d19261b`, merged; raeen-hle 461 green,
    kyty-graphics 477 green). All 83 measured-missing `libSceAgc` NIDs
    registered: 63 real (KytyPS5 ports — faithful 14-DW `CbBranch`, rewinds,
    workload markers, packet-patch families, 32 GetSize pinned to paired
    writers/Mesa PM4), 11 `register_incomplete` honest-error (arg order or
    semantics unreferenced in any license-compatible source — never guessed
    packets). MEASURED: GTA V render-path unresolved **83 → 0**; total 265 →
    88 (all online/dialog/input = item 3 territory). Attribution updated
    (THIRD_PARTY_NOTICES: KytyPS5 + Mesa; reference-port-ledger).
  - [~] **B. Compute queue in the PM4 processor.** WAVE 2 IN PROGRESS
    (gpu-pipeline agent, worktree, 2026-07-27). Phase B plan (from A):
    (1) execute submitted ACBs — port KytyPS5 `GraphicsRunSubmitCompute`
    incl. the 5-DW ACB descriptor indirection (magic 0x5533ccaa) in
    `submit_acb`; (2) graphics↔compute ordering — port
    `flush_pending_graphics_segment_before_acb` (agc.cpp:3698-3839), matching
    ACB R_WAIT_MEM waits against pending RELEASE_MEM producers; integrate at
    the cross-queue label latch in `raeen-gpu/src/agc_exec.rs`; (3)
    queue-indexed execution context for DISPATCH/ACQUIRE_MEM/RELEASE_MEM/
    WAIT/COND_EXEC arms already encoded in `kyty-graphics/src/run.rs`.
  - [~] **C. `libSceAmpr` (46 NIDs)** — WAVE 2 IN PROGRESS (hle-stubber
    agent, bundled with the abort/exit noreturn fix, 2026-07-27). Honest
    semantics (likely synchronous completion).
  - [ ] **D. Re-measure GTA V** stop point; update
    `docs/gta5-blocker-analysis-2026-07-27.md`.
- **Why:** One structural capability (async compute) unblocks the whole AAA
  class, not just GTA V.
- **Where:** `crates/raeen-hle/src/libsce_agc.rs`, `crates/kyty-graphics/`
  (run.rs, gen5), `reference/kyty` + `reference/kytyps5` (study/port, MIT).

### 3. Tier-B offline stubs for online/social/service libs
- [x] DONE 2026-07-27 (commit `f88a39b`, merged; raeen-hle 481 green
  post-merge with ACB + UE5 batches). ~134 functions: NpWebApi2 (+17,
  context-not-found/not-signed-in), Voice (15, real port handles, silence),
  NpManager/NpAuth (+13, async completes immediately SIGNED_OUT), AvPlayer
  (+10, real handle, immediate EOS so video waits never hang; decode =
  explicit future work), dialogs (24: Ime USER_CANCELED, WebBrowser,
  NpCommerce — no purchase ever granted), online misc (34: remoteplay/
  shareplay/streaming/content/VideoRecordingP incl. 1 anonymous NID), tail
  sweep (19 incl. real `sceHttpUriEscape` + `sceRtcGetTime_t`). Partial
  behavior uses `register_incomplete` so coverage tooling can't mistake
  resolved for working. Http2 needed nothing (already covered — measured).
- **Measured after items 2A+3:** remaining unresolved surface is essentially
  Tier C: 46 `libSceAmpr` + `_Ctype` (the 83 AGC are now registered).
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
- [x] `hle_stack_chk_fail` no longer returns (commit `d818df9`, merged
  `9091b17`): actionable error report (thread, guest RA, HLE ring, stack code
  chain), EOWNERDEAD mutex release, unwinds via `request_exit(0xa002_0006)` on
  both dispatch paths; runtime acceptance test proves the old stub executed a
  poison tail and the fix doesn't.
- [x] getdents/fstat layout audit vs shadPS4 (same commit). ROOT CAUSE FOUND:
  we returned one 512-byte record per call with `d_reclen = 512` (real Orbis
  dirent = 264) — guests copying by d_reclen smashed their own canary. Now
  FreeBSD/shadPS4 packed-record semantics (align4(8+namlen+1), block-packed,
  0 at EOF, dot entries, byte-offset cursor); fstat on dir fds now S_IFDIR
  with shadPS4 values. Empty dir: 0x200 once, then 0. Verified-not-assumed
  list in the ledger. Post-merge: raeen-hle 445, kernel 43+2, runtime
  77+47+1 green.
- [~] NEW (residual risk from the audit): `hle_abort` and `hle_exit` still
  log-and-return-0 — same noreturn hazard when a title's fatal path uses
  `abort()`. Route both through the same `request_exit` unwind. WAVE 2 IN
  PROGRESS (hle-stubber agent, bundled with item 2C Ampr, 2026-07-27).
- [ ] Re-test Until Dawn live: expect it past /app0/deepfiles now; capture the
  next stop point if any.
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
- [~] WAVE 2 IN PROGRESS (general agent, worktree, 2026-07-27).
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
- [x] DONE (shell-ui agent, worktree, 2026-07-27). One shareable file per
  faulted session: `logs/crashes/<title-id>_<UTC>.report.md` — fault site
  (module+offset+RIP bytes), recent HLE calls per thread, unresolved-NID
  inventory, GPU counters, host info, paired `.dmp`/log paths. Rich report
  written by the runner child (`--run-eboot` fault path); Shell writes a
  fallback only when a minidump landed without one. Shell view: Settings ▸
  System lists recent reports (Confirm opens; Open Folder / Copy to
  Clipboard action rows). Core is pure + unit-tested
  (`crates/raeen-gui/src/crash_report.rs`).
- **Where:** `crates/raeen-gui` (view), runtime fault plumbing
  (d87765a crash reporting groundwork).

### 13. Clippy debt: kyty-graphics recompile.rs (~91 clippy-1.97 lints)
- [~] WAVE 2 IN PROGRESS (general agent, worktree, 2026-07-27). Includes the
  `is_multiple_of` MSRV lint at analysis.rs:3195 flagged by the ACB agent.
  Note: wave-1 agents ran clippy successfully in worktrees, so AppControl may
  only block some exes — try locally, CI remains the gate of record.

---

## P2 — Performance

### 14. Async pipeline compilation
- [ ] Compile new Vulkan pipelines off the render thread; skip-draw (or
  fallback) until ready; persistent shader cache already exists — this kills
  first-encounter hitching. Measure with the perf HUD (item 15).
- **Where:** `crates/kyty-graphics` pipeline creation path, `raeen-gpu`.

### 15. In-Shell perf HUD
- [x] CODE DONE 2026-07-27 (commit `c6f4ed6`, merged `8a31528`; raeen-gui 185
  green post-merge). F3 toggle + Settings ▸ Advanced row; FrameTimeStats
  120-sample window off published-frame epochs; painter+rects top-right HUD
  (FPS, avg/worst frame ms, flip FPS, upload/drain/fence/read/sRGB ms); no
  puffin dependency; persisted as `general.perf_hud`.
- [ ] LIVE VERIFY pending (verify skill, real title): F3 on/off over Minecraft,
  plausible numbers, Settings row tracks, HUD absent on fault overlay,
  portrait-display capture gotcha.
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
- [x] CODE DONE 2026-07-27 (commit `c6f4ed6`, merged `8a31528`). F12 anywhere +
  pad Create rising edge in-session (press still forwarded to guest); dumps
  published guest frame to `screenshots/<id>_<UTC>.png` (image crate, refuses
  non-RGBA8/truncated); toasts; `screenshots/` gitignored. 4 tests + config
  persistence test.
- [ ] LIVE VERIFY pending: F12/Create toast + PNG on disk during a session;
  no-session info toast on Home.

### 19. DualSense passthrough
- [ ] Rumble first (gilrs ff or hidapi), then haptics/adaptive triggers via
  DualSense output reports. Un-reserve the Settings toggle ("DualSense
  Features" currently labeled reserved — see 2026-07-27 settings-backend
  audit).

### 20. Auto-updater
- [x] LANDED VIA PARALLEL SESSION (Codex, commit `0e77a65` lineage):
  `crates/raeen-gui/src/updater.rs` — release parsing (rejects plain-HTTP,
  bad JSON/tags), swap script (waits, swaps, relaunches, self-deletes),
  Inno Setup installer assets. Tests green in the 185-test raeen-gui suite.
- [ ] Verify end-to-end against a real GitHub release when one exists.

### 21. Video clip capture (stretch, after 18)
- [ ] Rolling ring of recent frames → mp4 on hotkey. Only after screenshot +
  perf work; consider `re_mp4`/ffmpeg-sidecar licensing (GPL-2.0-only tree).

---

## Session log (append per session)

- **2026-07-27 (Fable 5):** Checklist created. Delegated in parallel
  (worktree-isolated agents): item 1 → milestone-driver, item 2A →
  gpu-pipeline, item 3 → hle-stubber, item 4 → hle-stubber, items 15+18 →
  shell-ui. **All five landed and merged same day**: M5 CLOSED (`f56132f`),
  ACB phase A (`d19261b`, GTA V render-path unresolved 83→0), Tier-B batch
  (`f88a39b`, ~134 fns), UE5 root-cause fixes (`d818df9`, d_reclen=512 canary
  smash found), perf HUD + screenshots (`c6f4ed6`). Item 20 (updater) found
  already done by a parallel Codex session. Full workspace suite green after
  all merges (55 suites, 0 failures; one transient shared-target/ build race
  with the parallel session, passed on retry). NEXT highest-value: item 2B
  (compute-queue execution) + live re-measure of GTA V and Until Dawn, item
  4's abort/exit noreturn fix, live-verify pass for HUD/screenshots.
