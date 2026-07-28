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
- [x] Dragon Ball fixed and live A/B verified (2026-07-28): clean-room local
  disassembly proved module+0x241c820 is the title's downstream frame-pointer
  backtrace collector, not the source bug. Raeen was fabricating
  `ETIMEDOUT` after 50 ms for a NULL-timeout `sceKernelWaitEqueue`, sending
  AGC workers into that fatal path. NULL waits now remain indefinite through
  bounded host slices; finite waits keep their real deadline, bad timeout
  pointers return `EFAULT`, and user/VideoOut/APR/AGC producers wake the
  shared wait service. The exact previously crashing image ran 20 s past its
  ~7 s failure window with 0 guest faults, 0 equeue timeouts, and 4 AGC
  submission markers. `raeen-hle`: 525/525 tests and clippy all-targets
  `-D warnings` green.
- **Where:** `crates/raeen-hle/` (stack_chk, getdents/fstat),
  `crates/raeen-kernel/` VFS, ledger ITEM 2 notes.

### 5. Ordered GPU side effects — steps 3–5
- [x] Step 3 (code): unified timestamp clock behind `RAEEN_UNIFIED_GPU_CLOCK`
  (default OFF, bit-identical): `raeen_gpu::gpu_clock` is the one authority;
  HLE `next_gpu_timestamp` delegates under the gate, the worker CP gets an
  injectable `set_timestamp_source` that declines when the gate is off.
- [x] Step 3 (live A/B, 2026-07-28): on ASTRO.BOT, unified clock alone
  matched the default path at the current blocker (4 flips, 20 shader errors,
  17 GPU errors, same mixed 3D/2D Rgba16f storage-image crash at 28.1 s vs
  29.8 s baseline). It did not reintroduce the timestamp-fence park, but
  raised peak RAM from 2.55 to 3.14 GiB, so the gate remains default-OFF.
- [x] Step 4: events/EOP execute in-stream under
  `RAEEN_DEFER_GPU_SIDE_EFFECTS` (CP `SideEffect` records →
  `raeen_gpu::ordered_side_effects` queue → HLE drains at submit /
  WaitEqueue poll / VideoOut status); eager duplicates gated off. Dual-policy
  tests in kyty-graphics + raeen-gpu + raeen-hle.
- [x] Step 5: flip-pending ordered the same way; CP test pins that a flip
  behind an unmet wait is not recorded (so never delivered) until the wait
  genuinely passes; VideoOut status reads drain worker flips.
- [x] Live A/B of `RAEEN_DEFER_GPU_SIDE_EFFECTS=1` on Minecraft + ASTRO
  (2026-07-28). Minecraft reached a correctly rendered menu at 32 FPS but
  booted slower in the earlier contention-heavy run (~85 s vs ~50 s).
  ASTRO improved from a mixed-image host crash at 4 flips/~30 s to a clean
  bounded timeout at 36 flips/60 s. Enabling both gates also reached 36 flips
  but caused 94 shader-error retries vs 18 with deferred-only. These are
  measured A/B results, not a default-on acceptance: both gates stay OFF
  pending the mixed-storage fix and a quiet Minecraft performance re-run.
- **Context:** Steps 0–2 landed (gate + fail-open + IT_DMA_DATA in-stream,
  `cp_op_it_dma_data` in kyty-graphics run.rs). Design notes in ledger
  2026-07-25/26 ITEM 4; steps 3–5 implementation in ledger 2026-07-27.

- **A/B LIVE (2026-07-28 ~11:45pm):** `RAEEN_DEFER_GPU_SIDE_EFFECTS=1` on
  Minecraft (release build w/ all merges): reaches the full menu with correct
  rendering at 32 FPS; boot slower (~85 s vs ~50 s gate-off) — consistent with
  the bounded event-delivery latency, but measured under heavy CPU contention
  (9 agent builds running), so re-A/B on a quiet machine before considering
  default-ON. No hang, no visual corruption. `RAEEN_UNIFIED_GPU_CLOCK` A/B on
  ASTRO.BOT deferred until the machine is quiet (timing-sensitive).

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
- [~] HARNESS DONE 2026-07-28 (worktree agent; xtask 55/55 green = baseline 25
  + 30 new; fmt + `clippy -p xtask --all-targets -D warnings` green). LIVE
  30-min run remains — main session, needs the local game install.
  `cargo xtask soak [--game ID] [--minutes N (30)] [--exe PATH] [--input
  none|SPEC|FILE] [--stall-secs N (10)] [--boot-secs N (180)]`: launches the
  prebuilt exe (`--run-eboot`, baseline's staleness gate) and monitors LIVE:
  epoch = high-water of `WORKER TIMING flips=` / `total_flips=` +
  `sceVideoOutSubmitFlip` lines (log tailed incrementally); process-tree
  CPU/mem via sysinfo. FAILS on: no epoch advance > stall limit (armed after
  first advance; boot has its own budget), the pthread_sync
  `scePthreadMutexLock stuck >3s — deadlock` warning (mutex/owner parsed),
  or exit before deadline — prints frozen-window timestamps + 80-line log
  tail, writes `artifacts/soak/<run-id>/report.txt`, exits nonzero. Success
  prints min/avg/max window FPS, overall flips/s, worst stall, peak mem,
  avg/peak CPU. Synthetic input EXISTS and is wired: `--input` forwards a
  validated `raeen-input` replay spec via `RAEEN_INPUT_SCRIPT` +
  `RAEEN_RUNNER_CHILD=1` (script outranks Shell IPC/native pads; final
  snapshot holds forever, so `…:ls_up` walks for the rest of the soak).
  `--input none` (default) still catches frozen frames/deadlocks but only on
  the boot/idle path — reduced coverage, stated in the report.
- [ ] LIVE: `cargo build --release -p raeen-gui` then
  `cargo xtask soak --game minecraft --minutes 30` (add `--input` for
  in-world coverage). Human-supervised run closes this.
- [ ] Re-verify the in-world hang that did NOT reproduce post-fix (mutex
  `0x1019a1d48c0` / `0x1019a1d32e0`); if it recurs, instrument holder call
  path via `RAEEN_TRACE_HLE` + owner name. (The soak fails loudly on that
  exact warning line now.)

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
- [x] DONE 2026-07-27 (shell-ui agent, worktree; raeen-gui 202 green — 195
  baseline + 7 new). New `crates/raeen-gui/src/compat.rs`: mirrors the
  xtask baseline schema (source of truth `xtask/src/schema.rs`; stage order
  mirrors `baseline.rs::stage_rank`), loads `artifacts/compat/latest.json`
  (missing/malformed → no badges, never a crash), folds in the Shell's own
  session ledger newest-wins (a newer fault → Broken "last session"; a newer
  clean exit only ever *upgrades* to Boots — it can't disprove Playable).
  Mapping: Refused/Detected/Launching/Crashed → Broken; ran-but-no-flips →
  Boots; flips + render errors or fps<10 → Menu; clean flips fps≥10 →
  In-game; fps≥30 → Playable; no data → Untested (no chip drawn, no noise).
  Tile chip is painter-drawn at explicit rects (dark pill + theme-derived
  status dot via `compat::badge_color` — semantic hue, theme accent's S/L);
  per-game overlay shows badge + provenance on the title row. Matching:
  `param.json` title id first, display title fallback. Badges refresh on
  rescan and session exit; baseline index loads once per Shell start.
- **Where:** `crates/raeen-gui/src/compat.rs`,
  `crates/raeen-gui/src/shell/{mod,home,per_game}.rs`.

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
- [x] CODE DONE 2026-07-28 (worktree branch, this commit). Three layers:
  1. **Register decode complete** (kyty-graphics): every `CB_COLOR{0-7}`
     sub-register now lands in a named `RenderTarget` field — VIEW, ATTRIB,
     DCC_CONTROL, CMASK(+SLICE), FMASK(+SLICE), CLEAR_WORD0/1, DCC_BASE,
     plus the Gen5 `BASE/CMASK/FMASK/DCC_BASE_EXT` high-byte blocks
     (0x390–0x3AF) and the full INFO field set. Compression metadata is
     decoded but deliberately NOT emulated — one process-wide INFO note
     replaces the per-register "unknown context register" warnings.
  2. **MRT1–7 actually attach** (raeen-gpu): `DrawState.mrt` carries slots
     1–7 (per-slot format, `CB_TARGET_MASK` nibble write mask,
     `CB_BLEND{n}_CONTROL` blend); the offscreen pipeline declares N colour
     attachments (`independentBlend` enabled when supported, identical-state
     fallback otherwise), seeds each extra from the framebuffer map, and
     reads every attachment back to its own guest base. MRT draws take the
     immediate path (deferred batch can't file multi-target readbacks).
  3. **Fast clear** (shadPS4 FilterDraw port): `CB_COLOR_CONTROL.MODE 2`
     (eliminate-fast-clear) is consumed and applied as a real direct clear —
     the packed CLEAR_WORD splatted over the target's framebuffer entry +
     persistent-image eviction; modes 3/5/6 (resolve/decompress) are named
     once-logged skips, never silent scene draws.
  - Limitation (named): guest pixel shaders exporting `mrt1`+ still fail
    translation (recompiler handles MRT0 exports only), so real-title MRT
    output waits on the shader-side `exp mrt1-7` extension — plumbing and
    tests are in place for it.
  - Tests: kyty-graphics 484 (register decode incl. every slot/family);
    raeen-gpu lib 275 (MRT state translation, fast-clear decode + session
    FCE end-to-end) + `tests/mrt_targets.rs` (2 iron Vulkan MRT draws:
    dual-output FS hits both attachments; per-attachment write mask + LOAD
    seed verified pixel-exact).

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
- [x] RUMBLE DONE 2026-07-28 (worktree agent): guest → host → hardware.
  `scePadSetVibration` implemented for real (2-byte `{large, small}` param,
  shadPS4/SharpEmu-checked) → `OrbisKernel::set_pad_rumble` (seq bumps every
  call) → frame-IPC rumble word at header offset 104 (child → Shell, reverse
  of the pad channel; no VERSION bump — lives in zeroed padding, degrades to
  no-rumble across mismatched builds) → Shell `RumbleRouter`
  (`raeen_input::rumble`) → hardware. Output path is **direct HID output
  reports for the DualSense** (SharpEmu port in `hid.rs`: USB 0x02 / BT 0x31
  + CRC-32, dedicated writer thread, second device handle) **+
  `XInputSetState`** for Xbox-class pads — NOT gilrs ff (gilrs's Windows
  backend is XInput anyway; direct calls avoid the effect-object lifecycle
  and cover the raw-HID DualSense gilrs never sees). Settings ▸ Controllers ▸
  DualSense Features un-reserved: ON routes vibration, OFF drops it, applies
  live, persists. Safety: motors stop when the session ends/quits, and a 5 s
  no-refresh auto-stop covers a guest that never clears (real firmware +
  shadPS4 persist indefinitely; titles refresh far more often, and every
  scePadSetVibration call — same values or not — refreshes the deadline).
  Tests: rumble router/wire 6, HID output reports 3, HLE SetVibration 2,
  frame-IPC round-trip 1, config persistence 1.
- [ ] LIVE VERIFY pending (needs the user's controller): launch a rumbling
  title with a DualSense over USB, then BT, then an Xbox pad; toggle
  DualSense Features OFF mid-rumble (must stop); quit to Shell mid-rumble
  (must stop).
- [ ] Haptics / adaptive triggers / lightbar via full DualSense output
  reports = later (scePadSetTriggerEffect is still validate-and-ack).

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

## FINAL WAVE OUTCOME (2026-07-28, after "stop all agents and commit to main")

MERGED to main (tests + clippy green workspace-wide, 2373 tests, pushed):
soak harness (item 9 tooling, `cargo xtask soak`), compatibility badges
(item 11), MRT1-7 + fast-clear (item 16, commit 68b166e), trophy store
(item 17), DualSense rumble (item 19, commit dd295f0). Merge fixups:
per_game::draw carries BOTH badge and trophies params; 8-arg overlay allow.

UNMERGED WIP branches (committed on their worktree branches, NOT on main):
- `worktree-agent-af052f8a8fffa3490` — item 14 async pipelines. Lib tests
  275/275 green on its branch, but it refactored pipeline creation into a
  GraphicsPipelineRecipe that predates the MRT merge — structural conflict
  in offscreen.rs/draw_translate.rs. NEEDS: rebase onto main + extend the
  recipe with the MRT attachment/blend lists, then merge.
- `worktree-agent-a2ef7b7d8f3338947` — item 21 clips, mid settings-row work.
- `worktree-agent-af9b1e5e4a3652f06` — item 22 ASTRO fix, mid compute.rs
  refactor (per-binding storage-image dims).
- `worktree-agent-a5fe061b18bfa5dca` — item 6 phase 2, early (design in the
  branch + this file's history).


---

# ROUND 2 (added 2026-07-28) — next improvements, features, and crate adoptions

Numbering continues from item 22. Same conventions. Grounded in tonight's
findings: the measured blockers (canary smash, ASTRO mixed-dim shaders, MRT
shader exports), the live-verify experience, and a workspace-dep audit
(every crate below is absent from Cargo.toml today; licenses checked
GPL-2.0-compatible).

## R2-P0 — Engine critical path

### 23. Shader `exp mrt1-7` recompiler extension
- [ ] The MRT pipeline side landed (item 16) but the shader recompiler still
  handles only MRT0 exports — real-title MRT output needs exp targets
  0x01–0x07 declared as `%outColor1..7`. Named by the item-16 agent as THE
  follow-up on the GTA V-class critical path.
- **Where:** `crates/kyty-graphics/src/shader/parse.rs` (exp target decode),
  `recompile.rs` + `spirv.rs` (per-target output variables + writes).
- **Acceptance:** fixture shader exporting to mrt0+mrt1 → validated SPIR-V
  with two outputs; iron test via `tests/mrt_targets.rs` extension.

### 24. Hardware-watchpoint canary hunter (unblocks item 22a)
- [ ] The GTA V/Until Dawn canary smashes need to be caught IN THE ACT: arm
  x64 debug registers (DR0–DR3, write-watch on the canary slot / smashed
  region) on guest threads via SetThreadContext, report the smashing RIP +
  HLE context through the existing fault path. Env: `RAEEN_WATCH_ADDR=0x...`
  (+ optional auto-arm on `fs:0x28` canary of the faulting thread).
- **Why:** turns the two remaining title blockers from log-forensics into a
  one-run diagnosis. Also generally useful forever.
- **Where:** `crates/raeen-runtime` (thread context, VEH single-step arm),
  runner CLI plumbing.

### 25. AvPlayer real playback — audio first (menus/cutscenes)
- [ ] Tier-B AvPlayer returns immediate EOS (honest, but menus that gate on
  video completion show nothing). Real path: demux MP4 (`mp4` crate, MIT) +
  decode the AUDIO track via `symphonia` (verify exact license before
  adding; pure Rust AAC/MP3) through the existing audio mixer; video frames
  stay honest-black with correct PTS pacing and real EOS timing. H.264
  video decode stays future (no license-clean pure-Rust decoder; openh264
  pulls a Cisco binary — refuse).
- **Where:** `crates/raeen-hle/src/libsce_media.rs` + a small
  `raeen-media` helper or module.

### 26. GPU device-fault capture in crash reports
- [ ] Enable `VK_EXT_device_fault` (when present) + debug-utils labels on
  submissions; on VK_ERROR_DEVICE_LOST, append the fault report + last label
  to the crash report (item 12's assembly). Directly serves ASTRO-class GPU
  crashes.
- **Where:** `crates/raeen-gpu/src/vulkan/instance.rs` + crash_report.rs.

## R2-P1 — Tooling / CI / hygiene

### 27. cargo-deny in CI (license + advisory gate)
- [ ] `deny.toml`: allowlist GPL-2.0-compatible licenses, forbid
  copyleft-incompatible + unknown; RustSec advisories; duplicate-version
  bans as warn. This mechanizes the clean-room license policy that is
  currently manual — highest-value single CI addition for this repo.

### 28. cargo-nextest as the test runner (local + CI)
- [ ] Faster parallel runs, per-test timeouts, AND automatic retries with
  flake detection — directly addresses the vblank flake class. Add a
  CI profile with retries=2; annotate known-slow tests rather than
  deleting them.

### 29. Fix the vblank flake properly
- [ ] `consecutive_vblank_waits_land_one_period_apart` depends on wall-clock
  timing and fails under load (three separate agents hit it tonight).
  Inject a mock/monotonic test clock into the vblank source instead.
  (`crates/raeen-hle/src/libsce_video_out.rs`.)

### 30. Diagnostics bundle (one-click bug report)
- [ ] Settings ▸ System ▸ "Export Diagnostics": zip (crate `zip`, MIT) the
  newest crash report + paired .dmp + log tail + sysinfo block + effective
  config with user paths redacted → `logs/diagnostics-<UTC>.zip`, toast +
  open-folder. Pairs with the public release for actionable user reports.

### 31. Local minidump symbolization
- [ ] Resolve OUR OWN frames in captured .dmp files: `minidump` +
  `minidump-processor` + `pdb` crates (MIT/Apache) against the build's PDB;
  append a "host stack (symbolized)" section to crash reports. Requires
  release.yml to also upload the PDB as a release asset.

### 32. Coverage + dep hygiene (nice-to-have)
- [ ] `cargo llvm-cov` job (informational), `cargo-machete` unused-dep sweep.

## R2-P2 — Performance

### 33. mimalloc global allocator (measured evaluation)
- [ ] Windows heap is a known cost on allocation-heavy paths. Add `mimalloc`
  (MIT) behind a cargo feature, benchmark boot time + in-world FPS +
  shader-translate time on Minecraft before/after (perf HUD + criterion),
  adopt only with numbers.

### 34. Precise frame pacing with spin_sleep
- [ ] The frame limiter's sleep granularity on Windows is timer-quantum
  dependent (~15 ms); `spin_sleep` (MIT) gives hybrid sleep+spin. Wire into
  the existing limiter + vsync paths; verify pacing variance drops via the
  HUD worst-frame number.

### 35. Perf HUD graphs
- [ ] `egui_plot` (MIT/Apache): frame-time sparkline + 1%-low readout on the
  F3 HUD (second row). The numbers already exist (FrameTimeStats).

### 36. Micro-opt passes with evidence (smallvec / memchr / half)
- [ ] Only with criterion deltas: `smallvec` for PM4 packet arg vectors,
  `memchr` in log/NID scanning hot paths, `half` for f16 texture/format
  conversions currently hand-rolled. Skip any that do not measure.

### 37. profiling facade + optional Tracy
- [ ] Adopt the `profiling` crate as the annotation layer (keeps puffin as a
  backend, adds `tracing-tracy` as an opt-in feature) — frame-timeline
  debugging beyond puffin's flamegraph when chasing pacing/submission bugs.

## R2-P3 — Features / product

### 38. Save-data manager
- [ ] Per-game overlay: Export Save (zip of the title's savedata dir with a
  manifest), Import Save (validate title id, back up the old one first),
  Open Save Folder. Uses the same `zip` dep as item 30.

### 39. Title updates / patch PKGs
- [ ] Install user-supplied title-update PKGs: version-aware overlay mount in
  the VFS (patch files shadow base files), Shell shows installed version vs
  base. PKG parse exists in raeen-loader; the overlay logic is new.

### 40. Multiple local users
- [ ] User picker on the Shell (PS5-style), per-user savedata roots +
  `sceUserService*` returning the selected user; trophies (item 17 store)
  become per-user.

### 41. Input remapping + multi-controller
- [ ] Settings ▸ Controllers: rebind keyboard→pad and pad→pad buttons
  (persisted per input device), P2+ controller assignment (gilrs ids →
  pad handles) for local multiplayer titles.

### 42. Accessibility + i18n groundwork
- [ ] UI scale slider (egui zoom), colorblind-safe badge palette (distinct
  hue + shape per level), and string-table extraction for the Shell (ship
  en; structure for community translations).

### 43. Compat table publishing to the website
- [ ] `cargo xtask compat publish` already emits a sanitized markdown table —
  wire a manual (not scheduled) flow to copy it into the raeen-site repo
  (separate repo; downloads/compat pages go live once releases are public).

## R2-P4 — Portability (north star step 2)

### 44. Linux groundwork
- [ ] Non-Windows `GuestArena` via mmap (MAP_FIXED_NOREPLACE identity map),
  SIGSEGV-based trap-and-emulate equivalent of VEH, FSGSBASE via
  `arch_prctl`. First gate: `cargo check --target x86_64-unknown-linux-gnu`
  green in CI (compile-only), then the M1 fixture suite on Linux CI. Keep
  `#[cfg]` honest per the existing gotcha.

## R2 crate summary (all verified absent from Cargo.toml, licenses OK)

| Crate | License | For item |
|-------|---------|----------|
| `zip` | MIT | 30, 38 |
| `minidump`, `minidump-processor`, `pdb` | MIT/Apache | 31 |
| `mimalloc` | MIT | 33 |
| `spin_sleep` | MIT | 34 |
| `egui_plot` | MIT/Apache | 35 |
| `smallvec`, `memchr`, `half` | MIT/Apache | 36 |
| `profiling`, `tracing-tracy` (opt-in) | MIT/Apache | 37 |
| `mp4`, `symphonia` (license-verify first) | MIT / mixed | 25 |
| `blake3` | CC0/Apache-2.0 | shader-cache/content hashing (bench vs sha1 first) |
| cargo-deny, cargo-nextest, cargo-llvm-cov, cargo-machete | tools, MIT/Apache | 27, 28, 32 |
