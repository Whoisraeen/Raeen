# Raeen — Codex battle-ready workflow (2026-07-25)

Execution plan for Codex CLI to take Raeen from *"one of eight tested titles
renders"* to *"handles an unknown PS5 title without falling over, and tells you
exactly why when it does."*

**Read this doc before `AGENTS.md`.** `AGENTS.md` is stale (see Phase 0, task 1).

---

## 0. How to use this doc

- Phases are **ordered and resumable**. Do not start phase N+1 until phase N's
  exit gate is green.
- Every task names its **files**, its **port source**, and its **exit gate**.
- Rank numbers (`recon #7`) refer to `docs/reference-recon-roadmap.md`, which is
  the most accurate roadmap in the tree. Trust it over
  `docs/rendering-blockers-and-port-plan-2026-07-22.md`.
- Work on `main`. Commit only when the user asks.

---

## 1. Ground truth — what is actually true today

Several in-tree docs are stale and will misdirect you. This section overrides
them.

### Measured compatibility (build `df38544`, `compat/COMPATIBILITY.md`)

| Title | Stage | Wall | Flips | First blocker |
|---|---|---:|---:|---|
| Minecraft | TimedOut | 180.3s | 2048 | none observed |
| ASTRO.BOT | TimedOut | 180.2s | **0** | none observed |
| Until Dawn | Exited | 4.3s | 0 | guest fault `read 0xa` |
| Subnautica Below Zero | Exited | 1.9s | 0 | unimplemented `libScePad` import |

Also known from the ledger, not yet in the table: GTA V faults after 2733 HLE
calls; Avatar hits `libScePlayGoDialog` unimplemented; Dragon Ball shares Until
Dawn's `read 0xa` signature; A Plague Tale dies on a **host** divide-by-zero
(`0xC0000094`) — that one is Raeen's bug, not the title's.

### Corrections to stale docs — do not re-do closed work

| Stale claim | Where | Reality |
|---|---|---|
| "M1 walls are the critical path" | `AGENTS.md`, `docs/homebrew-gap-analysis.md` | M1 closed long ago. The emulator boots commercial titles. |
| "`texture_vk_format` has no BC format arms" | `rendering-blockers-…-07-22.md` | **Done.** BC1–BC7 at `draw_translate.rs:1020-1033`. |
| "No depth attachment ever, `depth: None` hard-coded" | `rendering-blockers-…-07-22.md` | **Done.** D24/D32 formats at `draw_translate.rs:331-333`; depth render target wired. |
| "`tiling.rs` is CPU-only, 2 modes" | `rendering-blockers-…-07-22.md` | 4 modes (5/9/24/27), verified bit-identical to SharpEmu. Still CPU-only; modes 1/4/8 still missing. |
| "A swapchain is the fps fix" | folklore | **False.** `docs/reference-recon-roadmap.md` measured it: swapchain is recon #18, *large*, and **not** the lever. See Phase 2. |

### Confirmed current gaps

- **No Vulkan swapchain at all** — zero `SwapchainKHR` in `raeen-gpu` or
  `kyty-graphics`. Present path is GPU → CPU readback → egui texture upload
  (`crates/raeen-gui/src/shell/present.rs:14-17`). This is a real ceiling but
  **not** the current binding constraint.
- **Blocking fence waits** — `wait_for_fences(…, u64::MAX)` at
  `vulkan/compute.rs:1592`, `vulkan/offscreen.rs:1518` and `:4131`.
- **~1044 NID registrations** against a console surface of tens of thousands.
- `xtask/src/nids.rs` and `crates/raeen-firmware/examples/hunt_nid_names.rs` are
  **untracked** — unfinished NID tooling. Finish them in Phase 1.
- `crates/raeen-firmware/src/dynlib/nid_names.txt` has **185,088** lines of
  NID→name data. Underused.
- **LLE works.** Shipped-libc mspace runs LLE by default
  (`raeen-firmware/src/lib.rs:1672`). This matters for Phase 5.

---

## 2. Prime directive and honesty rules

Raeen's whole discipline is *never claim a capability you have not measured*.
Codex must hold this line — it is the reason this codebase's ledger is
trustworthy.

1. **Never report "works on every title."** Report per-title measured results
   with build SHA, wall time, flip count, and first blocker. One title booting
   is evidence about one title.
2. **Never mark a milestone (M0–M5) closed** without its acceptance test. Read
   `.agents/skills/acceptance-gate` first.
3. **Synthetic fixtures are not proof.** A hand-built module that exercises only
   what the pipeline already supports proves nothing about real titles.
4. **Name the miss.** Every refusal, skip, or unsupported path must log *what*
   it refused and *why*, rate-limited. Silent skips are how days get lost.
5. **Record negative results.** "Tried X, measured no change" is as valuable as
   a fix and belongs in `.superpowers/sdd/progress.md`.
6. **Clean-room is absolute.** No Sony keys, SDK, firmware, or game bytes in
   tree. Reference clones stay under gitignored `reference/`. Read
   `.agents/skills/clean-room` before any port. Attribute in
   `THIRD_PARTY_NOTICES.md` and log in `docs/reference-port-ledger.md`.

---

## 3. Standing rules — violate these and you will regress the tree

### Minecraft is the regression gate

Minecraft (PPSA17221) is the one reliably-rendering title. It is the canary for
every change.

- **Before claiming any task done:** 180s release run, compare flips/s and wedge
  behavior against baseline (2048 flips / menu 38-42 FPS). A wedge or flip-rate
  drop = **revert**.
- **Fire-and-forget flip flush is a KNOWN Minecraft killer.** Tried and reverted
  2026-07-20: flips stop ~10s in, main thread deadlocks on a title mutex held by
  the flipping render-pool thread. Cause never diagnosed. **Every Phase 2
  present-path change must be env-gated and default-OFF** until that deadlock is
  understood.
- **Do not casually change vblank pacer semantics.** Minecraft paces its entire
  animation loop on `sceVideoOutWaitVblank`.
- **PM4 additions must be additive-only** — execute packets currently skipped by
  length; never change behavior of packets already executing. Minecraft's DCBs
  are proven byte-exact against the current register tables.

### Build gotchas that have cost real time

- **A running `raeen.exe` blocks `cargo build`** with `Access is denied`, *and*
  `cargo build … | tail` reports tail's exit code (0), so the failure is
  invisible and the next run silently measures a **stale binary**. Redirect to a
  file and capture `$?`. `tasklist` name-filtering misses holders that a module
  scan (`Get-Process | %{ $_.Modules }`) finds.
- **Use a separate `CARGO_TARGET_DIR`** (e.g. `../Raeen-target-dev`) so
  development never touches the binary the user is running.
- **Two concurrent cargo builds on one `target/` deadlock.** Isolate parallel
  work in git worktrees.
- **Verify staleness by process StartTime vs binary mtime**, not by the log
  (it rotates per launch).

### Measurement protocol

Every perf or compat claim needs: build SHA, release build, `cargo xtask compat
run --tier all --timeout 180`, and the resulting JSON saved to scratch. Never
compare a debug run to a release baseline.

---

## Phase 0 — Instrumentation (days 1-3)

**Goal:** stop the one-bug-per-build-cycle grind. This phase is the force
multiplier; everything after runs 5-10x faster because of it.

### 0.1 Fix `AGENTS.md` — do this first

It is a stale copy of an old `CLAUDE.md`. It claims M1 (crt0 stack, TLS relocs,
printf observability) is the critical path — all shipped. It references
`.Codex/skills/` and `.Codex/agents/`, which do not exist; the real paths are
`.agents/skills/` and `.codex/agents/`. Rewrite it from the current `CLAUDE.md`
plus §1 of this doc.

**Exit gate:** a cold Codex session reading `AGENTS.md` correctly identifies the
current gate as M4-class title compatibility, not M1.

### 0.2 Fail-soft unknown NIDs — the single highest-leverage change

Today an unimplemented import **kills the process**
(`raeen-runtime/src/lib.rs:322`). Subnautica dies at 1.9s having reported
exactly one missing function.

Change the default to: log the NID + resolved name + calling module, return 0,
continue. Keep the hard-fail behind `RAEEN_STRICT_NIDS=1` for debugging.

**Why this dominates everything:** one run now yields a *complete inventory* of
every missing import in a title instead of one. Subnautica's ledger trajectory —
`GetAgeLevel` → `GetAccessibilityChatTranscription` →
`scePadDeviceClassGetExtendedInformation` — was three full build/measure/run
cycles for three functions. This collapses that to one run.

**Exit gate:** Subnautica runs >30s and emits a deduplicated list of every
unimplemented NID it touched.

### 0.3 Readable crash reports (recon #1)

Logs are ~92% call-trace-ring spam. Collapse the 4096-entry ring into one DEBUG
event; lead with a distilled ERROR crash report (faulting address, module +
offset, last N distinct HLE calls, thread identity).

**Files:** `raeen-runtime/src/dispatch.rs`, `logging.rs`.
**Port from:** shadPS4's one-record-per-event model.

### 0.4 Per-flip timing HUD (recon #2)

Always-on, cheap. Frame time broken down by: GPU worker drain, fence wait,
readback, sRGB encode, egui upload. **Measure before optimizing** — Phase 2
depends on this existing.

**Files:** `raeen-gpu/src/vulkan/offscreen.rs`, `agc_exec.rs`,
`raeen-gui/src/shell/present.rs`.

### 0.5 Close the VFS sandbox escape (recon #3)

Windows drive-letter/symlink escape in `crates/raeen-kernel/src/filesystem/mod.rs`.
Fail closed, canonicalize and contain. **This is a live security bug.**
Port from SharpEmu `e01092a`.

### 0.6 Free acquisitions

- Clone **Mesa** for `src/amd/addrlib/` (MIT — authoritative tiling),
  `src/amd/registers/*.json` (machine-readable RDNA2 register DB),
  `src/amd/common/sid.h` (`PKT3_*` packet defines).
- Install **Ghidra** + a PS5 PRX loader. Phase 4 depends on it.
- Add Mesa to `compat/reference-state.json` and `THIRD_PARTY_NOTICES.md`.

> MIT and GPL-2.0 both flow into GPL-2.0-only. **LLVM's AMDGPU backend is
> Apache-2.0-with-LLVM-exception — NOT GPL-2.0-only compatible. Do not copy it.**
> `llvm-mc` as an external differential-testing oracle copies nothing and is fine.

### Phase 0 exit gate

One 180s run per title produces (a) a complete missing-NID inventory, (b) a
readable one-screen crash report, (c) a per-flip timing breakdown. Minecraft
A/B unchanged.

---

## Phase 1 — HLE breadth (week 1)

**Goal:** no title dies on an unimplemented import in its first 30 seconds.

### 1.1 Harvest the inventory

Run all 8 titles with Phase 0.2 in place. Collect every missing NID at once.

### 1.2 Bulk stub generation — never one-at-a-time

Finish `xtask/src/nids.rs` and
`crates/raeen-firmware/examples/hunt_nid_names.rs` (both untracked). Drive from
`nid_names.txt` (185k lines), `reference/ps4libdoc/known_names.txt`, and
`reference/ps5-payload-sdk`'s `prospero-nid`.

**Register by family, not by fault.** The ledger already learned this — the
accessibility getters were "registered as a FAMILY rather than one-per-fault"
because each miss otherwise costs a whole measure/build/run cycle.

Emit a table of *registered but not implemented* so coverage is never mistaken
for correctness. (Kyty Gen5 carries 119 `EXIT_NOT_IMPLEMENTED`; that is the
trap to avoid.)

### 1.3 Targeted stubs from recon

- **#5** — `sceAgc *Cb/*Dcb GetSize` sizing probes (size in `rax`, no writes).
  SharpEmu `74a5198`.
- **#6** — `libSceAjm` `Batch{Initialize,JobDecode,StartBuffer,Wait,Cancel}`
  silence stubs. SharpEmu `2272b9b`/`d3600c9`.
- `libScePlayGoDialog` (blocks Avatar), `scePadDeviceClassGetExtendedInformation`
  (blocks Subnautica).

### 1.4 Load-path speed (recon #14, #15, #16, #26)

~27s of ASTRO's load is the VFS read hot path. Rework: no handle clone, no
double copy, no global write lock, read directly into guest VA. Add an on-disk
cache of the decrypted+linked+patched image keyed by input hash + build version
(~800ms cold → tens of ms warm). Prefilter the syscall-patch instruction decode
(`memchr 0F 05`). One-pass stub-eligible NID collection.

**Exit gate:** 6/8 titles run >30s without an unimplemented-import death.
Cold-launch load time for ASTRO cut by >50%. Minecraft A/B clean.

---

## Phase 2 — The FPS lever (week 2)

**Goal:** Minecraft holds 60 FPS in-world; ASTRO produces non-zero flips.

**Measured status (2026-08-03): green.** ASTRO.BOT previously produced 96
flips and bounded async flip passed its 3×180 s no-wedge gate. Minecraft's
strict scripted run
`scratch/mc-phase2-strict-30m/soak-1785724463876` lasted 30m00.6s, produced
133,280 flips (74.1 overall; 74.7 average telemetry-window FPS), had a 2.0 s
worst no-flip window and zero deadlock warnings. Across 3,853 steady
post-transition 32-frame windows, p50/p95/p99 frame time was
13.1/14.7/16.4 ms and the derived 1% low was 61.0 FPS. This closes Phase 2;
it does not establish compatibility or performance for unmeasured titles.

Read `docs/reference-recon-roadmap.md` §"The one thing to internalize" before
starting. The measured diagnosis:

> ASTRO runs at ~0.4 fps because `sceVideoOutSubmitFlip → present_scanout →
> consume_flush(wait:true)` **blocks the guest synchronously behind ALL queued
> DCB submits drained by a SINGLE GPU worker** — each flip inherits the full
> worker-drain + fence-stall latency. The 60 Hz pacer already works.

### 2.1 Bounded fire-and-forget flip (recon #7) — THE lever

Frames-in-flight semaphore, cap 1-2, deadlock-aware.

**This is the known Minecraft killer.** Env-gate it (`RAEEN_ASYNC_FLIP=1`),
default OFF, and diagnose the 2026-07-20 deadlock *before* flipping the default:
main thread blocked on a title mutex held by the flipping render-pool thread.
Use Phase 0.3's crash reports plus the existing "stuck >3s" detector.

### 2.2 Command buffer ring (recon #12)

N=8 command buffers with per-buffer fences. Kills the per-submit
`wait_for_fences(u64::MAX)` round trip at `compute.rs:1592`,
`offscreen.rs:1518`/`:4131`. Port model: KytyPS5 `commandScheduler.cpp`
(deferred flush, one submit per frame/window). **Redesign Raeen-native — do not
transliterate.**

### 2.3 Drift-compensated vblank pacing (recon #13)

Waitable timer + accumulator. Port from shadPS4 `AccurateSleep`/`Timer`. Keep
the blocking-call contract for `sceVideoOutWaitVblank`; keep 60 Hz default.

### 2.4 Cheap wins

- **#27** — cheaper guest-mem present pass + sticky flip-miss fallback.
- **#28** — serialize `VkPipelineCache` to disk (cold-launch stutter).
- Cache the per-flip HDR→sRGB re-encode by (fb base, generation). Currently
  8.3M px × 3 `powf` every flip, uncached.

**Exit gate:** ASTRO flips > 0. Minecraft ≥ 60 FPS in-world, no wedge across 3
consecutive 180s runs. Timing HUD shows fence wait no longer dominant.

---

## Phase 3 — Make 3D actually draw (weeks 3-4)

**Goal:** titles that currently render nothing render *something*.

### 3.1 Null-descriptor fallback (recon #8) — biggest graphics item

Move invalid descriptors from **translate-time refuse** to **draw-time null
fallback**. Today one bad descriptor kills an entire draw, silently. This is
why titles render black rather than rendering wrong — and wrong is far more
debuggable than black. `robustness2` is already enabled.

**Port from:** shadPS4 `GetSharp` / `Null()`.

### 3.2 Texture correctness

- **#9** — read sampled mip 0 from the GFX10 mip-chain **tail** offset. Raeen
  always reads `t.base40()`, so any `last_level > 0` texture samples from the
  wrong offset. SharpEmu `6ee445f`.
- **#10** — sample 2D-array / cube / 3D textures with real layers.
  SharpEmu `25d741b`.
- **#4** — accept padded source row pitch (`bufferRowLength`) in guest image
  uploads. SharpEmu `0ae785c`.
- **AddrLib transcription** from Mesa — replaces guessed swizzle equations with
  authoritative tables; closes tile modes 1/4/8. The existing rate-limited
  `(tile_mode, format)` refusal diagnostic tells you which modes titles actually
  bind — **let it gate the work**.
- Derive `VkImageType` from the T# type nibble rather than `depth > 1`. A
  type-10 T# with `depth()==0` currently gets SPIR-V `Dim3D` with a 2D image —
  the mismatch class blamed for an ASTRO device loss.

### 3.3 MRT (recon #11)

Bind all enabled render targets via `render_target_mask`. Port from Kyty
`GraphicsRender.cpp`.

> Note: a `note_active_color_slots` diagnostic fired **zero** times across a
> full Minecraft run — Minecraft is single-target. Do not size this work off
> Minecraft; validate against ASTRO or a UE5 title.

### 3.4 PM4 holes — packets emitted but never executed

`IT_DISPATCH_INDIRECT` (0x16), `IT_COPY_DATA` (0x40, includes reference-clock
writes Raeen has nothing for), `IT_INDIRECT_BUFFER` (0x3f),
`IT_SET_PREDICATION` (0x20) / `IT_COND_EXEC` (0x22), `IT_GET_LOD_STATS` (0x8e),
and the `R_VS`/`R_PS` full stage binds (without which `vs_regs.data_addr` stays
0 and **every draw is skipped** for packet-form titles).

**Port from:** KytyPS5 `pm4Handlers.cpp`, `graphicsRun.cpp`. Additive-only.

### 3.5 Ordered GPU side effects

Raeen applies DMA_DATA / WRITE_DATA / RELEASE_MEM / EVENT_WRITE **eagerly at
submit** (`libsce_agc.rs:1180-1290`), so labels claim completion of GPU work
that has not run, and `sceVideoOutIsFlipPending` always returns 0. Enqueue them
on the GPU queue so they apply after the draws preceding them in PM4 order.
Adopt SharpEmu's fail-open on side-effect faults (Raeen returns
`INVALID_ARGUMENT` at `:1188`).

**Port from:** SharpEmu `AgcExports.cs:3726-3778` `SubmitOrderedGpuSideEffect`.
Preserve the flip rendezvous semantics — see standing rules.

**Exit gate:** 3-4 titles reach a rendered menu. Frame dumps
(`RAEEN_DUMP_FRAMES`) show recognizable content, not flat fills. Minecraft
visually unchanged.

---

## Phase 4 — Multi-title root causes (week 5)

**Goal:** fix bugs that unblock *classes* of titles, not single titles.

### 4.1 The UE5 `read 0xa` fault — highest value in the tree

Until Dawn and Dragon Ball, **identical signature**, both after
`sceKernelWaitEqueue` ETIMEDOUT. One root cause, two titles, and UE5 is a large
slice of the PS5 library.

Load the eboot in Ghidra (Phase 0.6) and read the faulting site directly. The
ledger's ASTRO work — `+0xe03f1a`, `r14 = 0xAAAAAAAC` allocator poison, walking
the voice list by hand from register dumps — shows what this costs *without* a
disassembler. Do not repeat that.

### 4.2 A Plague Tale host divide-by-zero

`0xC0000094 STATUS_INTEGER_DIVIDE_BY_ZERO` is a **host** crash — Raeen's bug.
Any title can hit it.

### 4.3 ASTRO's zero flips

A rendering title producing no frames in 180s is its own class. Use the Phase
0.4 HUD and Phase 3.4 PM4 execution to find where the frame dies.

### 4.4 Minecraft's blank post-menu page

cohtml renders the menu fine, so the engine works; the post-press view loads no
content. Documented multi-session RE. Next lead is **not** another HLE guess —
diff which cohtml exports fire before vs after the press
(`RAEEN_TRAP_MODULE_EXPORTS=cohtml`), or dump the cohtml View's loaded URL.

**Exit gate:** UE5 fault root-caused with a named mechanism (not a workaround).
6/8 titles past their `df38544` first blocker.

---

## Phase 5 — Scale beyond the 8-title corpus (week 6+)

**Goal:** the pipeline handles *unknown* titles gracefully. This is what
"battle ready" can actually mean as a testable condition.

### 5.1 Automated compat sweep

Extend `cargo xtask compat run` to a larger corpus, run unattended, auto-update
`compat/COMPATIBILITY.md`, and diff per-title stage/flips/first-blocker against
the previous build. **Regression detection is the deliverable**, not the title
count.

### 5.2 Graceful degradation as a design rule

Audit every hard-failure path for a degrade-and-log alternative:
unknown NID (Phase 0.2), unsupported tile mode, unsupported format, unknown PM4
packet, invalid descriptor (Phase 3.1), missing module. **An unknown title
should produce a diagnostic, not a crash.** This is the single property that
generalizes to titles nobody has tested.

### 5.3 LLE scale-out — the structural unlock

The LLE path already works (shipped-libc mspace runs LLE by default). With
user-supplied decrypted system `.prx` modules, the NID problem changes shape:
load Sony's own implementations instead of hand-writing thousands of HLE
functions, and HLE only the kernel boundary.

Codex should make the LLE path **robust and preferred where a real module is
available**, with per-family HLE override (the existing `force_hle_nid`
mechanism) for families where LLE is measured worse.

**The modules themselves are user-supplied and must never enter the repo.**
Codex builds the road; the user decides whether to drive on it.

### 5.4 Deferred — large items, explicitly out of scope until Phase 5 lands

Recon #18 real swapchain, #19 dedicated present/vblank thread, #20 page-fault
dirty tracking, #21 descriptor dataflow tracing over flattened SRT/EUD,
#22 async compute rings, #23 Bink FMV via FFmpeg, #29 MSAA, #30 GPU-EOP flips.
All marked **large**. Touching them early costs the wins above and delivers
nothing complete.

---

## 4. Exit gate summary

| Phase | Exit gate | Falsifiable by |
|---|---|---|
| 0 | Full missing-NID inventory + readable crash report + timing HUD, in one run per title | Run Subnautica; count distinct NIDs reported |
| 1 | 6/8 titles survive >30s without unimplemented-import death; ASTRO load −50% | `cargo xtask compat run --tier all` |
| 2 | ASTRO flips > 0; Minecraft ≥60 FPS in-world, no wedge in 3×180s | Compat JSON + timing HUD |
| 3 | 3-4 titles reach a rendered menu; frame dumps show real content | `RAEEN_DUMP_FRAMES` visual review |
| 4 | UE5 `read 0xa` root-caused with a named mechanism | Until Dawn + Dragon Ball both past it |
| 5 | Unknown-title regression sweep runs unattended and diffs builds | Add an untested title; it degrades, not crashes |

---

## 5. What "battle ready for any PS5 game" means operationally

The literal condition — every PS5 title playable — is not reachable on a
weeks-scale plan, and Codex must not report it as achieved. For calibration:
shadPS4 targets a simpler, better-documented console with dozens of contributors
and years of work, and covers a fraction of the PS4 library. Raeen is at 1 of 8
titles rendering. Per-title compatibility work is unbounded and continues
indefinitely.

What **is** reachable, and what this plan delivers, is the property that makes
an emulator battle-ready in the engineering sense:

> **No unknown input causes a hard failure, and every degradation names itself.**

An unknown title should boot as far as its actual gaps allow, log precisely
which NIDs, packets, formats, and tile modes it needed that Raeen lacks, and
hand the user a report specific enough to act on. That is testable, it is
achievable in the phases above, and it converts every future title from a
debugging expedition into a work item.

Phases 0-2 are the ones that compound. Do them in order.
