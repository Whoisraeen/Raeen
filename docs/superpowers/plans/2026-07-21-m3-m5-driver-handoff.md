# M3–M5 Driver Handoff — Full Plan + Prompt

**Date:** 2026-07-21
**For:** the next long-running driving agent (Opus-class)
**Mission:** each installed title showing its real graphics and playing like it
would on PS5 — M3 (interactive), M4 (commercial 2D title to menu), M5 (3D
title recognizable frames), then playability and 120 FPS.
**Supersedes:** `2026-07-20-shader-to-playable.md` (its *method* was validated
end-to-end; its checklist is ~70% executed; its "DMA is the composite gate"
hypothesis was measured FALSE — see Queue Coordination below).

---

## 0. The prompt (give this to the driving agent verbatim)

> You are driving Raeen (clean-room PS5 emulator, Rust, Windows) to M3–M5:
> every installed title rendering real graphics and playing. Work on `main`.
> Read, in order: `CLAUDE.md`, this file,
> `~/.claude/.../memory/MEMORY.md` index + the per-title memory bodies
> (astro-bot-boot-state, minecraft-boot-state, untildawn-dragonball-boot-state,
> per-title-graphics-import-profiles, provider-aware-hle-resolution,
> test-the-shell-not-just-the-cli), and `.superpowers/sdd/progress.md`.
> Then run THE LOOP (§2) against the highest-value title wall (§4). Fix one
> named blocker class per iteration with a regression test, verify (§6),
> commit per round, update the ledger and the per-title memory. Never claim a
> milestone without its acceptance test (`.claude/skills/acceptance-gate`).
> Never claim a title "renders" without the frame-dump + clear-colour protocol
> (§5). Delegate parallelizable batches to subagents with disjoint crate
> scopes (§7). Prefer porting from references (SharpEmu/Kyty/shadPS4 — GPL-2
> compatible; NEVER GPL-3 code) over inventing semantics; cite file:line in
> doc comments. When a wall resists two rounds, run the reference emulator
> itself against the same title and mine its runtime behavior (§8 — this
> broke the biggest wall of 2026-07-20/21).

---

## 1. Current state (all measured; commits on `main` through `50e36e1`+)

| Milestone | Status |
|---|---|
| M0, M1, M2 | CLOSED (acceptance-tested; do not re-prove) |
| M3 | Pieces exist (flips, vblank events, pad HLE); formal gate unclaimed; no true swapchain |
| M4 | Minecraft is the candidate — blocked ONLY on PSN auth stub (§4.2) |
| M5 | ASTRO.BOT renders verified title pixels; scene gated on queue coordination (§4.1) |

Per title (fresh measurements, not aspirations):

- **ASTRO.BOT** — boots fault-free to its full render loop (~2,097 draws,
  ~2,925 dispatches, 73+ flips per run; 518 compute storage writebacks).
  Presents an evolving fullscreen gradient (its pre-scene pass) — verified
  title pixels, NOT the emulator clear color. 5 unique untranslatable compute
  shaders remain (named). One CS quarantined for GPU device-loss
  (`0x5006c5f00`, `RAEEN_SKIP_CS` forensics exist).
- **Minecraft** — boots deep (LevelDB/Gameface/RakNet), flips at 40–63/s
  after perf stages A–C, stalls ~90s in on PSN auth
  (`SceNpAuthAuthorizedAppDialog`, `SceNpWebApi`). Menu HTML
  (`data/gui/dist/hbui`) never loads. GPU is NOT its blocker.
- **Until Dawn / Dragon Ball (UE5 pair)** — identical graphics import sets,
  now fully resolved (zero MISSING). Blocked pre-RHI on condvar starvation
  (engine init handshake; stack chains recovered in the ledger). PREDICTED
  next wall once unblocked: the CP does not execute `IT_INDIRECT_BUFFER`
  (0x3F) chain packets (`DcbJump`/`CbBranch` emit them).
- **A Plague Tale** — never booted; smallest gap; all graphics imports now
  resolve; multi-submit is real (u32-stride-4 size array, SharpEmu-verified).

Perf (goal: 120 FPS at least one title; `/goal` hook may still be active):
stage A (pipeline/image caches — build 11 µs/draw), stage B (deferred
per-flush readback — submit p50 74 µs), stage C (flush-per-flip 1.02/flip,
flip-limited readback p50 7 ms, upload ring) ALL COMMITTED. Vblank pacing is
epoch-anchored with `RAEEN_VBLANK_HZ` (default 60). Minecraft boot animation
self-paces ~60; sustained-120 evidence needs an unthrottled phase (= the menu,
= the PSN stub). True zero-copy swapchain assessed: needs
`VK_KHR_external_memory_win32` interop or shared device with eframe/wgpu —
M3-sized project, documented in `shell/present.rs`.

**In flight at handoff:** one agent porting three SharpEmu mechanisms
(§4.1 items 1–3). If its work is uncommitted in the tree, verify (§6) and
commit if green; its report is authoritative over this doc for those items.

## 2. THE LOOP (validated over ~10 rounds; do not improvise around it)

1. Bounded release run: `cargo build --release -p raeen-gui` then
   `RAEEN_LOG="warn,raeen_gpu=debug" RAEEN_DUMP_FRAMES=<dir> timeout -s KILL 300
   ./target/release/raeen.exe --run-eboot "Games/<title>/<app>/eboot.bin" > run.log 2>&1`
2. Classify the FIRST named blocker by count:
   `grep -oE "next_gen: [a-z_0-9]+: [^\"]{0,60}" run.log | sed 's/at addr.*//' | sort | uniq -c | sort -rn | head`
   plus draw skips, dispatch skips, writeback counts, unimplemented imports.
3. Fix ONLY that class (enum+parse+recompile+naga-test for opcodes; reference-
   cited semantics for HLE; named refusal if genuinely unportable).
4. Verify (§6), re-measure, commit with the measured before/after in the
   message, update ledger + memory.
5. Success metric ladder: translate_failed → dispatch/draw skips → writebacks
   → frame histogram (§5) → recognizable frame → interactive → FPS.

## 3. Traps that cost sessions (all learned the hard way)

- Logs filter via `RAEEN_LOG`, **not** RUST_LOG. Frame dumps via
  `RAEEN_DUMP_FRAMES`. Validation opt-in `RAEEN_VULKAN_VALIDATION=1`
  (~0.9 s/pipeline — looks like a hang).
- **Shell ≠ CLI** (three recorded divergences). Verify user-facing claims in
  the real Shell (`.claude/skills/verify`); the title-VA window reservation
  now runs first in `main` — keep it first.
- **Clear-colour protocol** before any "it renders" claim (§5).
- **Provider-aware NIDs**: register under the library the title imports from;
  `--imports <eboot> [lib]` shows MISSING and HLE(other-lib) rows;
  `--missing-nids` for the full sweep. Implement on measured call for
  non-graphics libs; graphics sets may be closed statically per engine.
- Read the per-title memory BODY before diagnosing — they are mostly
  eliminated hypotheses.
- No python/magick on this box: convert PPM via PowerShell System.Drawing
  (recipe in astro-bot memory).
- Known flake: parallel two-device ash tests (shader_fetch / debug-utils) —
  rerun solo before believing a failure.
- Windows sleep quantization (~15.6 ms) breaks naive frame pacing — the
  vblank waiter's coarse-sleep+yield-spin pattern is the template.
- `scratch/` holds captured shader dumps + disasm (`shader_probe` example);
  `RAEEN_DUMP_SHADERS` + `enumerate_dumps` for fresh captures.

## 4. Per-title drive plans (highest value first)

### 4.1 ASTRO.BOT → M5 (recognizable frame, then play)
1. **Queue coordination — THE scene gate** (both we and SharpEmu stall here):
   ACB compute queues park in `WAIT_REG_MEM` waits whose producers are other
   queues' memory writes. Port SharpEmu's suspend/resume
   (AgcExports.cs:4508-4529): suspend buffer (resume offset + remaining
   dwords), re-check after every completed work item's writebacks, resume on
   genuine label write. NEVER force-satisfy. (In flight at handoff.)
2. **EUD/SRT as memory**: replace captured-descriptor refusals with dispatch-
   time guest-memory snapshot SSBOs (256 KiB–16 MiB windows, bounds-checked,
   degrade-to-zero) per Gen5ShaderScalarEvaluator.cs:1836-2052. Kills the
   last analysis-refusal class. (In flight.)
3. **Unquarantine the scene writer** (`0x5006c5f00`): synthesized default
   sampler for sampler-less sampled bindings (Presenter :6314/8121), LDS
   index masking (:2007), CPU-resolved runtime T# indices (dynamic → skip+log
   :654), optional SPIR-V contract validation (:9721). (In flight.)
4. Then: 5 remaining untranslatable CS by name; CB format `0x3` with a full
   attachment+blend+export plan (device-loss history — keep the reject until
   solved); `IT_COPY_DATA` when measured.
5. Acceptance: frame dump passes §5 with recognizable content →
   claim M5 entry with the screenshot. Then pad input → frame change = M3
   interactive evidence.

### 4.2 Minecraft → M4 (menu) and the 120 FPS proof
⚠️ CORRECTION (minecraft-boot-state memory body, line ~240): the "PSN auth
stall" is a DISPROVEN misread — the title never CALLS those NIDs (0
unimplemented-import faults, 0 thread deaths); the `SceNp*` strings are the
LINKER naming missing imports, not the game invoking PSN. Do NOT stub PSN to
"reach the menu" — that lead is dead. The real wall (memory body): a LIVE-LOCK
— after resource-pack enumeration + savedata + RakNet init, every thread parks
in SHORT polling waits (each <3s, so no starvation detector trips) and no
predicate ever becomes true. Everyone polls; nothing advances. Main thread
waits in `pthread_cond_timedwait` on a cond nothing signals to completion.
1. Find the predicate: correlate the main thread's waited cond with which
   worker should make it true; dump the guest flag/counter it re-checks. The
   memory has ~10 eliminated hypotheses — READ THE BODY before diagnosing.
2. Gameface/Ore-UI page handoff (`data/gui/dist/hbui`) — HLE class, not GPU;
   only relevant AFTER the live-lock clears (menu HTML never opened yet).
3. When the menu runs unthrottled: `RAEEN_VBLANK_HZ=120` and measure
   flips/s sustained — this is the 120 FPS goal's test bench. If flip cost
   caps it, next lever is async flip readback (attempted once — deadlocked a
   title mutex; re-attempt condition documented at `present_scanout`).
4. Regression rule: any MUBUF/addressing change re-tests Minecraft.

### 4.3 UE5 pair → boot, then graphics
1. The condvar starvation: trace who should post the RHI-init events
   (ledger has stack chains, `RAEEN_TRACE_COND`). This is engine-handshake
   RE, not missing imports (all resolve now).
2. The moment they submit: implement CP `IT_INDIRECT_BUFFER` chain execution
   (predicted wall, run.rs has no handler; decode counts 0x25 not 0x24).

### 4.4 Plague Tale — first boot triage
Run it; walk faults with the fault-site reporter + `--resolve-got` +
`--find-calls`. Its multi-submit path is ready and tested.

### 4.5 120 FPS — MEASURED CEILING IS TITLE-CPU, NOT GRAPHICS (2026-07-21)
Do not chase the vblank/present path for 120 FPS — it is already fast enough.
Hard measurements (release, stages A–C + epoch vblank committed):
- **Present pipeline capability: >120 fps proven.** Min inter-flip interval
  2.03 ms (~490/s); `sceVideoOutSubmitFlip` has NO pacing sleep (records +
  returns); Minecraft never calls `sceVideoOutWaitVblank` (0 calls) so
  `RAEEN_VBLANK_HZ` is IRRELEVANT to it (it applies only to titles that park
  on vblank). The stage-C "vblank sleep was the ceiling" note was for the
  WaitVblank path; Minecraft is the SubmitFlip path and was never throttled
  by us.
- **ASTRO.BOT: ~4 fps, GPU/compute-bound** (present indices 64→128 spanned
  15.3 s). Heavy HDR dispatches + per-flush readback. Matches SharpEmu's
  heavy-3D-title territory. Not a 120 candidate.
- **Minecraft boot animation: steady ~43 fps, TITLE-guest-CPU-bound.**
  Inter-flip intervals cluster tightly at 22–23 ms (NOT bursty, NO cluster
  at 8 ms) = the title's own boot-animation loop costs ~22 ms of guest x86
  per frame. It never negotiates a 120 mode (`IsOutputSupported`/
  `ConfigureOutput` = 0 calls). Our GPU/present is not the cap.
- **CORRECTION (deeper measurement, RAEEN_TIME_DRAW on Minecraft):** the 22 ms
  frame is NOT purely title-CPU. Per-draw `build_us` p50 **969 µs** (p90 2.6 ms,
  max 30 ms) × ~20 draws/frame ≈ **~20 ms of OUR per-draw build throughput** on
  the GPU worker thread, which the guest render thread blocks behind at the
  synchronous flip flush (flush_us p50 only 2.2 ms — the flush is cheap; the
  BUILD is the cost). So a real graphics lever remains after all.
- **THE LEVER (stage D — top FPS work):** `Resources::build` (offscreen.rs
  ~1219) still allocates PER DRAW: `create_depth_target` + `create_depth_buffers`
  (a fresh VkImage + view + device memory every draw — Minecraft uses depth on
  all ~20 draws/frame), plus guest vertex/index buffer creation. Stage A's own
  report flagged "depth targets are still per-draw." FIX: make depth targets
  PERSISTENT keyed by `DB_Z_WRITE_BASE`, mirroring the committed persistent
  COLOR-target machinery in `vulkan/cache.rs` (loadOp=LOAD, dirty tracking).
  Then verify guest-buffer uploads actually hit the stage-C upload ring (pool)
  rather than create/destroy. Estimated: cutting build ~2.5× (≈969→≈400 µs/draw)
  takes Minecraft's boot animation from ~43 toward ~90 fps; combined with
  batching all a frame's draw command buffers into ONE vkQueueSubmit, 120 is
  reachable on the boot-animation phase alone (no gameplay needed).
  RISK: hot-path + a prior async-flip attempt deadlocked — change carefully,
  keep RAEEN_NO_DEFER as the A/B, re-test Minecraft + ASTRO each step.
- **CONCLUSION:** 120 FPS is reachable via stage D (persistent depth targets +
  submit batching) on Minecraft's boot-animation phase — a graphics-perf lever,
  measurable now, no title progression required. Secondary path: any title in a
  light gameplay phase. Do NOT fake the number (acceptance-gate); demonstrate
  only with a measured per-second flip bucket ≥120 on a title actually
  rendering.

## 5. Honesty protocols (non-negotiable)

- **Rendering claim**: dump frames; histogram multiple offsets
  (`od` triple-count recipe in memory); require multiple distinct pixel
  values, temporal change across frames, and NOT the clear color
  (offscreen.rs CLEAR_COLOR — flip it to [1,0,0,1] for a controlled test if
  in doubt). Splash (`pic0.png`) is system UI, never title rendering.
- **Milestone claim**: in-tree acceptance test + the skill checklist.
- **Perf claim**: measured flips/s buckets or TIME_DRAW percentiles, with the
  exact build/env named.

## 6. Verification gates (every round, before every commit)

```
cargo test -p <touched crates>       # then dependents if ABI moved
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
cargo build --release -p raeen-gui   # the artifact users run
```
Commit only when all pass. One logical round per commit, measured
before/after in the message. Never commit `reference/`, `Games/`,
`config.toml`, or another session's unrelated uncommitted files.

## 7. Subagent pattern that worked

Disjoint crate scopes per agent (kyty-graphics vs raeen-gpu vs raeen-hle),
explicit "do NOT touch X / do not run fmt --all", evidence-first briefs with
measured counts and file:line citations, TDD required, report format
specified. Integrate + measure + commit in the driver loop. 3–4 parallel
agents max; the driver owns the ledger and memory.

## 8. The reference-emulator play (the 2026-07-21 breakthrough method)

When a wall resists: BUILD AND RUN SharpEmu against the same title.
`.NET 10` via winget; `global.json` pins 10.0.103 — build from OUTSIDE the
tree: `dotnet build <abs path>\SharpEmu.CLI.csproj -c Release`; exe at
`reference/sharpemu/artifacts/bin/Release/net10.0/win-x64/SharpEmu.exe
<eboot> --log-level debug` (+`SHARPEMU_APP0_DIR`, `SHARPEMU_DUMP_VIDEOOUT=1`,
`SHARPEMU_TRACE_COMPUTE_SHADER_ADDRESS`, `SHARPEMU_SKIP_COMPUTE_CS`).
Correlate its runtime logs with its source, then port with citations.
Guest VAs match ours — cross-reference addresses directly. Known result:
ASTRO.BOT in SharpEmu shows splash only (we are AHEAD); its value is
mechanisms, not pixels.

## 9. Definition of done ("plays like PS5")

Per title: real graphics on screen (§5), pad input drives it, stable minutes
of play, saves round-trip, no hangs from stub subsystems (audio fail-soft),
honest FPS report. All five titles at menu-or-better, at least one 3D title
in gameplay, at least one title at 120 FPS sustained in an unthrottled phase.
