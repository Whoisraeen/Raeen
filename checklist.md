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
  - [x] **B. DONE 2026-07-27** (gpu-pipeline agent worktree; kyty-graphics
    480 green (+3), raeen-hle 486 green (+5), raeen-gpu 269 green (+1)).
    Reality vs plan: ACBs already executed on a dedicated compute CP with
    suspend/resume + cross-queue latch; what was missing and is now ported
    from KytyPS5: (1) the 5-DW ACB descriptor indirection (magic 0x5533ccaa)
    in Submit[Multi]Acbs — descriptor-form ACBs previously failed decode and
    were dropped; (2) `flush_pending_graphics_segment_before_acb` + the
    pending-segment tracker (DCB-submit re-track, `alloc_command_dwords`
    growth, state on `OrbisKernel`) — unsubmitted graphics producers now
    flush as a DCB before the waiting ACB, truncated to the awaited
    RELEASE_MEM; (3) IT_DISPATCH_INDIRECT execution (both forms) + SET_BASE
    shader-type split in run.rs. Ledger + THIRD_PARTY_NOTICES updated.
  - [x] **C. DONE 2026-07-27** (hle-stubber agent, worktree branch; raeen-hle
    492 green). All 46 measured-missing `libSceAmpr` NIDs registered from
    KytyPS5 `src/libs/libAmpr.cpp` behavior: 37 real (nop/marker inert
    records, gather/scatter file reads with REAL eager data movement +
    host-tracked stream continuation, WriteAddress_04_00 completion write,
    GetType/GetBufferBaseAddress, every MeasureCommandSize* pinned to its
    writer's exact advance) + 9 `register_incomplete` honest-parity (waits
    dropped/counters unmodeled/map window unmapped — KytyPS5 does the same,
    but ours are tagged so coverage keeps naming them). MEASURED via local
    `xtask nids coverage` re-run: libSceAmpr unresolved **46 → 0**.
  - [x] **D. RE-MEASURED 2026-07-28** (live `cargo xtask baseline run`,
    build 9c7cc30): **zero unresolved NIDs**; early UD2 assert GONE; stage
    `timed_out` (survives the full 180 s window) with **4 flip events**.
    New first blocker: a real guest `__stack_chk_fail` on thread 31, properly
    caught by the noreturn handler. Blocker doc updated with the re-measure
    section. Next: hunt the canary smash (same family as Until Dawn's).
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
- [x] DONE 2026-07-27 (hle-stubber agent, worktree branch): `hle_abort` and
  `hle_exit` no longer return into their (noreturn) call sites. `abort` →
  full actionable report (thread, guest RA, recent-HLE ring, stack code
  chain, EOWNERDEAD mutex release — shared helper factored from the
  stack-chk fix) + `request_exit(ABORT_EXIT_CODE 0xa002_0106)`, distinct
  from the canary-smash code. `exit(status)` → `request_process_exit(status)`
  + `request_exit(status)` carrying the guest's own status (defense-in-depth
  behind the VEH terminating-function intercept). Runtime poison-tail
  acceptance tests for both (execute.rs, 49 green) + 2 hle unit tests.
- [x] Re-tested Until Dawn live (2026-07-28 baseline run): still exits at
  ~6.7 s — the canary smash PERSISTS post-dirent-fix (so getdents was a bug
  but not this title's trigger, or another overflow path remains). Now
  properly reported with an actionable signature (`ad605ab2…`) instead of a
  silent UD2 walk. Follow-up folded into the canary hunt (item 22 / GTA V
  shares the same failure family).
- [ ] Dragon Ball: worker threads deref a count (rax=2 → 0x20) as a list
  pointer at module+0x241c820 after WaitEqueue timeouts — needs fault-region
  disassembly follow-through (started, see ledger 2026-07-25/26 ITEM 2).
- **Where:** `crates/raeen-hle/` (stack_chk, getdents/fstat),
  `crates/raeen-kernel/` VFS, ledger ITEM 2 notes.

### 5. Ordered GPU side effects — steps 3–5
- [x] Step 3 (code): unified timestamp clock behind `RAEEN_UNIFIED_GPU_CLOCK`
  (default OFF, bit-identical): `raeen_gpu::gpu_clock` is the one authority;
  HLE `next_gpu_timestamp` delegates under the gate, the worker CP gets an
  injectable `set_timestamp_source` that declines when the gate is off.
- [ ] Step 3 (live A/B): flip `RAEEN_UNIFIED_GPU_CLOCK=1` on a live title —
  ASTRO timestamp-fence regression territory; stays with the main session.
- [x] Step 4: events/EOP execute in-stream under
  `RAEEN_DEFER_GPU_SIDE_EFFECTS` (CP `SideEffect` records →
  `raeen_gpu::ordered_side_effects` queue → HLE drains at submit /
  WaitEqueue poll / VideoOut status); eager duplicates gated off. Dual-policy
  tests in kyty-graphics + raeen-gpu + raeen-hle.
- [x] Step 5: flip-pending ordered the same way; CP test pins that a flip
  behind an unmet wait is not recorded (so never delivered) until the wait
  genuinely passes; VideoOut status reads drain worker flips.
- [ ] Live A/B of `RAEEN_DEFER_GPU_SIDE_EFFECTS=1` (Minecraft + ASTRO) before
  making either gate the default — main session.
- **Context:** Steps 0–2 landed (gate + fail-open + IT_DMA_DATA in-stream,
  `cp_op_it_dma_data` in kyty-graphics run.rs). Design notes in ledger
  2026-07-25/26 ITEM 4; steps 3–5 implementation in ledger 2026-07-27.

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
- [x] MECHANISM DONE 2026-07-27 (worktree agent, commit `6ff4132` on branch
  `worktree-agent-a8a0f93f4ac611946`; raeen-runtime 77 lib + 56 execute
  (+6 new) + 1 m3 green, raeen-hle 500 (+3 new), raeen-firmware 125, fmt
  clean, clippy `-D warnings` green on both touched crates).
  `GuestCallScheduler::call_guest(entry, [u64;6]) -> Result<u64, GuestCallError>`
  — an HLE handler calls INTO guest code mid-call and gets its RAX, on the
  current guest thread. Design (full rationale on `ActiveContext::call_guest`
  in dispatch.rs): on the **VEH path** the handler already executes on the
  guest stack (vectored exceptions dispatch on the faulting thread's stack),
  so a plain native `sysv64` call runs the callback below the trapped frame
  with compiler-kept alignment — and the armed recovery context, TLS/FS
  re-arm, on-demand commit, and terminating arms all stay live, so nested
  HLE imports trap normally. Nesting bounded only by guest stack; depth 2
  (HLE→cb→HLE→cb) pinned by test. Unwind composition: callback faults →
  `Err(Faulted)`; `request_exit` under the callback (stack_chk/abort) →
  clean unwind, interrupted handler provably never resumes (poison-tail +
  resumed-flag tests). **Direct leaf gateway refuses loudly** (`Unsupported`
  + error log): its bridge re-bases RSP to a fixed host-stack top on every
  entry, so re-entry there would clobber live gateway frames —
  `direct_dispatchable` lists only never-re-enter imports; refusal pinned by
  test overriding `libc::strlen`. Interrupted handler's `active_hle` +
  `pending_guest_call` saved/restored around the callback.
- [~] Retire the blocked class: `qsort` DONE (real in-place heapsort over
  guest memory, comparator gets REAL element pointers, verified end-to-end:
  guest fixture sorts + guest-side order check + comparator call counter;
  ledger 2026-07-25's "qsort (needs sync guest callback)" skip retired).
  Remaining, now unblocked by the same mechanism: `atexit` chain (needs a
  hook in the terminating-function arm before the exit longjmp), module
  init/fini callbacks, VideoOut/GPU event callbacks.
- **Where:** `crates/raeen-runtime/src/dispatch.rs`
  (`ActiveContext::call_guest`, `direct_gateway_active`),
  `crates/raeen-hle/src/lib.rs` (trait + `GuestCallError`),
  `crates/raeen-hle/src/libc.rs` (`hle_qsort`),
  `crates/raeen-runtime/tests/execute.rs` (6 acceptance tests).

### 8. AIO infrastructure
- [x] DONE 2026-07-27 (hle-stubber agent, worktree branch). The 5 skipped
  kernel AIO NIDs identified from phase1 coverage (Until Dawn + Dragon Ball
  Sparking Zero import exactly: `sceKernelAioSubmitReadCommands`,
  `SubmitWriteCommands`, `WaitRequest`, `PollRequests`, `DeleteRequest`) —
  implemented REAL, plus the 7 sibling spellings (`*Multiple`,
  plural/singular wait/poll/cancel/delete forms) that share the machinery.
  Engine: `crates/raeen-kernel/src/aio.rs` — 2 lazy host worker threads over
  the SAME VFS descriptor table as the sync path; submit returns ids
  immediately; wait blocks on condvar with timeout; poll never blocks;
  cancel aborts not-yet-started (in-flight finish normally, per Orbis
  "PROCESSING = could not cancel"); delete retires final state into a
  bounded ring so late polls still answer. Workers NEVER touch guest
  memory: write payloads captured at submit, read completions staged host-
  side and delivered through `ctx.mem` when the guest drains via the API
  (`crates/raeen-hle/src/kernel_aio.rs`). Layouts re-derived from public C
  decls, cross-checked vs shadPS4 aio.h (no code ported). Tests: kernel
  53 (+10), hle 507 (+10 incl. struct-layout + delete-before-wait
  delivery); clippy/fmt green.

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
- [x] CODE DONE 2026-07-27 (worktree agent; xtask 25/25 green, 18 new tests;
  clippy/fmt green on the crate). `cargo xtask baseline run` (prebuilt-binary
  only, staleness check, per-game retry, parts + merged.json, latest.json
  only on full success) + `baseline diff <old> [new] [--strict]` (stage/
  exit/flip/fps + unresolved-NID deltas with newly-missing/-resolved lists;
  Evidence gains optional `unresolved_nids`, old reports round-trip).
  Merged to main (commit `c065b11`); 25/25 green post-merge. LIVE RUN
  VALIDATED 2026-07-28: full 9-title sweep completed (exit 0, latest.json
  replaced, per-title NID harvesting worked — all titles now report 0
  unresolved); `baseline diff` produced the regression report that caught
  the ASTRO.BOT crash (item 22). python script deleted per plan. Item fully
  closed; change status to [x].
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
- [x] DONE 2026-07-27. `cargo clippy --workspace --all-targets -- -D warnings`
  is GREEN end to end, verified locally (AppControl did not block this
  session). Agent commit `aebb8f3` (merged `71eaea7`): kyty-graphics 87→0
  (5 mechanical fixes + 2 module-scoped test-fixture allows with
  justification), raeen-gpu 8→0 (3 fixes + 5 justified allows for phase-1
  plumbing / pinned deprecated fixtures). Main-session commits `d030259`
  (analysis.rs MSRV lint) + `ba870f7` (last six: raeen-gui ×4, raeen-input
  ×1, raeen-hle test code ×2 from today's batches). Tests: kyty-graphics 477,
  raeen-gpu 308, gui 185, input 18, hle 481 — all green; fmt clean.

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
- [x] LIVE VERIFIED 2026-07-28 (real Minecraft session, PrintWindow captures):
  F3 ON shows the full HUD (70 FPS, avg 14.3/worst 21.4 ms, flips 34 FPS,
  upload 0.85 ms, drain/fence/read/sRGB strip, "F3 hides this HUD" hint);
  F3 OFF returns the plain badge — never both. Numbers plausible vs the
  measured upload baseline. NOTE for future verifies: PrintWindow(hwnd,hdc,0x3)
  captures the Shell even when occluded by the user's windows — prefer it
  over CopyFromScreen.
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
- [x] LIVE VERIFIED 2026-07-28: F12 on Home -> "No game running" info toast
  at TopLeft, no file. F12 in a Minecraft session -> success toast naming
  screenshots\Minecraft_20260728-013027-231.png; the PNG is the pure guest
  frame (no Shell UI/HUD baked in), decodes correctly. Pad Create not
  exercised (no physical controller connected during the verify) — covered
  by unit tests; re-check when a controller is present.

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
- [!] BLOCKED (external, 2026-07-27): end-to-end verification needs a real
  GitHub release; `api.github.com/repos/Whoisraeen/Raeen/releases` returns
  404 (private repo or no releases). Unit coverage (parse/swap-script) is
  green; verify e2e when the first public release is published.

### 22. NEW (2026-07-28): baseline-diff catches — canary hunt + ASTRO.BOT crash
- [ ] **Canary smash hunt (GTA V thread 31 + Until Dawn ~6.7 s):** both titles
  now die on a real guest `__stack_chk_fail` with actionable reports. The
  getdents d_reclen overflow is fixed, so another guest-visible struct/size
  mismatch remains (hunt the same way: audit the syscalls in the pre-canary
  log window against shadPS4 layouts). This is now GTA V's top blocker.
- [ ] **ASTRO.BOT regression (TimedOut → Crashed 0xC0000409):** ACB Phase B
  made previously-dropped descriptor-form compute submissions execute; ASTRO
  now reaches 20 shader errors (`storage_texture_dim_format: mixed storage
  image dims/formats in one shader (Three vs Two, Rgba16f)`) and then a HOST
  fail-fast (STATUS_STACK_BUFFER_OVERRUN) after 83 s — that's OUR bug, not
  the guest's: (a) support or gracefully refuse mixed-dim storage images in
  one shader; (b) find the host-side buffer overrun in the new dispatch path.
- **Note:** Minecraft FPS in this run (65.6 vs 72.2) was measured while five
  agent worktree builds were running — contaminated, not a regression signal.

### 21. Video clip capture (stretch, after 18)
- [ ] Rolling ring of recent frames → mp4 on hotkey. Only after screenshot +
  perf work; consider `re_mp4`/ffmpeg-sidecar licensing (GPL-2.0-only tree).

---

## Session log (append per session)

- **2026-07-27 (Fable 5, wave 2):** All five wave-2 items landed and merged:
  ACB Phase B (`8c1dc34` — descriptor-form ACBs were being silently dropped,
  now execute; pre-ACB graphics flush; DISPATCH_INDIRECT), Ampr 46→0 +
  abort/exit noreturn (`69ef513`), xtask baseline (`c065b11`), crash report
  view (`b37d041`), clippy debt (`aebb8f3` + main-session `d030259`/`ba870f7`
  — workspace --all-targets -D warnings GREEN). Full workspace: 2257 tests,
  0 failures. Engine-side GTA V NID surface now fully accounted for.
  REMAINING (need the user's machine or careful sequential A/B): items 2D
  (GTA V re-measure), 5 (side effects 3-5), 6 (swapchain phase 2), 7 (guest
  callbacks), 8 (AIO), 9 (soak), 11 (badges — baseline format now exists),
  14 (async pipelines), 16 (MRT/fast-clear), 17 (trophies), 19 (DualSense),
  21 (clips); live-verify passes for HUD/screenshots/crash view/Until Dawn.
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

## INTERRUPTED WAVE (2026-07-27 ~8:45pm CT) — resume after the usage-limit reset (11:20pm CT)

Eight agents died mid-work on the session usage limit. Their worktrees under
`.claude/worktrees/agent-<id>/` hold UNCOMMITTED partial work. To resume: send
each agent a message to continue (same agent id), or start fresh agents in the
EXISTING worktrees telling them to finish + commit. States at interruption:

| Item | Worktree agent id | Last known state (from its progress notes) |
|------|-------------------|---------------------------------------------|
| 22b ASTRO shader/overrun | af9b1e5e4a3652f06 | mid shader fix: `storage_image_coord_text` + `storage_descriptor_index_constant` (3 files modified) |
| 19 DualSense rumble | a25b1414c6cf305da | near done: "un-reserve the Settings hint" left (12 files) |
| 17 Trophy store | ab8451f5668e88c3c | Shell wiring: ActiveSession construction/helpers/tick/draw call site (8 files) |
| 14 Async pipelines | af052f8a8fffa3490 | lib tests green (275, +6 pool tests); Vulkan-device integration test left (8 files) |
| 16 MRT1-7 + fast-clear | a75ce852ccac64bf2 | tests rerunning; checklist+ledger update left (12 files) |
| 21 Video clips | a2ef7b7d8f3338947 | settings.rs row const/count/draw/tests left (6 files) |
| 9 Soak harness | a02e09e7762605222 | adding record_memory to ResourceStats + resource test (5 files) |
| 6 GPU present phase 2 | (worktree gone — design done, no code) | full design in its final notification; relaunch fresh with the design pasted |
| 11 Badges | aae9109900b0f7b09 | was waiting on cargo test -p raeen-gui -j 4 (5 files) |

Also owed after resume: live 30-min soak (9), A/B gate flips (5), canary hunt
(22a), badges/crash-view/phase-2 live verifies, DualSense hardware check,
v0.1.0 release-workflow result check (GitHub Actions).
