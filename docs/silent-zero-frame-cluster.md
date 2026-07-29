# The silent zero-frame cluster

Four measured titles run cleanly, present nothing, and log no error at all.
This is the largest single failure cluster in the library and the only one our
tooling could not see: every other diagnosis this project has made started from
reading an error line, and these produce none.

| Title | id | Stage | wall | cpu | Engine | Log ends |
|---|---|---|---|---|---|---|
| **Blasphemous II** | PPSA13580 | timed_out | 180.1 s | **3.2 s** | Unity / IL2CPP | +2.7 s |
| Subnautica Below Zero | PPSA02456 | timed_out | 180.1 s | **1.5 s** | Unity / IL2CPP | +1.0 s |
| Dragon Ball Sparking Zero | PPSA15210 | timed_out | 180.1 s | **8.6 s** | UE5 (`/app0/sparkingzero/`) | +1.5 s |
| Until Dawn | PPSA15421 | timed_out | 180.2 s | **179.6 s** | UE5 (`/app0/bates/`) | +1.6 s |

Evidence: `artifacts/compat/indies-first.json` (Blasphemous II),
`artifacts/compat/post-astrofix-validated.json` (the other three), raw logs
under `artifacts/compat/raw/baseline-1785285421268/` and
`.../baseline-1785282849471/`.

---

## 1. The headline finding is about the instrument, not the titles

**The frame path is invisible at the log level the compatibility harness runs
at.** Every conclusion previously drawn from "the log contains no VideoOut
activity" was unsound.

Proof, from the same measured batch: **Minecraft presented 13,440 flips at
75.5 FPS and its log also contains zero `sceVideoOut` lines.**

```
$ grep -c sceVideoOut artifacts/compat/raw/baseline-1785282849471/PPSA17221-05b59012cd4e.stdout.log
0        # ...for a run whose report records flip_events: 13440
```

The cause: the whole of `crates/raeen-hle/src/libsce_video_out.rs` logs through
`debug!` — `sceVideoOutOpen`, `sceVideoOutSubmitFlip`, buffer registration, the
lot. The harness (`xtask/src/main.rs`) never sets `RAEEN_LOG`, so the emulator
runs at its `info` default and drops all of it.

Two consequences, both now fixed:

1. **We could not tell "never opened a video-out handle" from "presented 31
   frames".** The harness derived `flip_events` from the worker telemetry
   window (emitted once per 32 completed presents), the AGC progress line
   (power-of-two sampled), or a count of `sceVideoOutSubmitFlip` DEBUG lines
   (always zero). A report reading `flip_events: 0` really meant *somewhere
   between 0 and 31*.
2. **A comprehensive stall observer already existed and was never armed.**
   `RAEEN_STALL_DUMP` (`crates/raeen-gui/src/main.rs:1511`) dumps, every 6 s:
   each thread's last five HLE calls, every guest thread's RIP resolved to
   `module+offset` (`raeen_runtime::sample_guest_rips`), per-thread wall-clock
   sinks, the HLE call each thread is *currently inside*, and a shallow host
   backtrace per thread. The compat harness sets `RAEEN_TIME_WORKER`,
   `RAEEN_COMPAT_RUN_ID`, `RAEEN_VBLANK_HZ`, `RAEEN_ASYNC_FLIP` and
   `RAEEN_CALL_STATS` — never `RAEEN_STALL_DUMP`. Four 180-second stalls were
   measured with the stall observer switched off.

### What was added

`crates/raeen-core/src/frame_path.rs` — eight ordered counters covering the
whole chain, plus the first-occurrence timestamp of each:

```
frame path: reached=dcb_submitted | videoout_open=1@812ms buffers_registered=2@840ms
flip_rate_set=1@840ms dcb_submitted=91@1204ms draws=0 dispatches=0
flips_submitted=0 frames_published=0
```

* **Off by default.** Disabled, `frame_path::record` is one relaxed
  `AtomicBool` load and a not-taken branch — no counter traffic, no clock read.
* **Enable with `RAEEN_FRAME_PATH=<seconds>`** (`1`/`on`/empty = 10 s, `0` =
  count but do not report periodically).
* **Periodic, not at-exit.** `xtask` resolves a stalled title with
  `child.kill()`, so an at-exit hook would never fire for exactly the titles
  this exists for. A reporter thread logs the summary on an interval, so the
  last line before a hard kill carries the chain state.
* Armed in `crates/raeen-gui/src/main.rs` immediately after logging init —
  *before* the `--run-eboot` dispatch, which is the path the harness uses and
  which returns long before the Shell's config bridging.

Instrumentation points:

| Stage | Site |
|---|---|
| `videoout_open` | `libsce_video_out.rs::hle_open` |
| `buffers_registered` | `hle_register_buffers`, `hle_register_buffers2` |
| `flip_rate_set` | `hle_set_flip_rate` |
| `dcb_submitted` | `raeen-gpu/src/agc_exec.rs::execute_dcb_cp` |
| `draws` | the three `draw_count` accrual sites in `agc_exec.rs` |
| `flips_submitted` | `hle_submit_flip` |
| `frames_published` | `agc_exec.rs::publish_completed_frame` (single funnel) |

The harness now sets `RAEEN_FRAME_PATH=15` on every compat run, folds
`flips_submitted` into `flip_events` (exact, replacing the sampled estimate),
and — when a run produced no frames and logged nothing louder — synthesizes a
`first_blocker` from the summary. A silent title stops reporting
`first_blocker: null`.

---

## 2. Where the chain stops: what is known, and what is not

**Verified from the artifacts:**

* No title in the cluster produced a *completed present* (no worker telemetry
  window in 180 s ⇒ fewer than 32 presents; almost certainly zero).
* All four die within ~3 s of `_start`, then log nothing for the remaining
  ~177 s.
* The cluster is **not one stall class**. CPU time splits it cleanly:
  * **Hard-blocked** (Blasphemous II 3.2 s, Subnautica 1.5 s, DBSZ 8.6 s of CPU
    across 180 s wall): parked in a host wait. Category (c) — blocked on a
    synchronization primitive that never fires.
  * **Spinning** (Until Dawn 179.6 s ≈ exactly one core pegged): a guest busy
    loop. A different bug.
* The cluster is **two engine pairs**, and each pair stops at an identical
  fingerprint:
  * *Unity/IL2CPP* (Blasphemous II, Subnautica): last three lines are
    `sceKernelDlsym(handle=0, symbol='scriptingGetMem') ENOENT` →
    `sceKernelCreateEventFlag "resumeEvent"` → guest pthread spawn → silence.
  * *UE5* (Until Dawn, DBSZ): last lines are two
    `sceAgcGetRegisterDefaults2: unsupported version 13 — using the legacy
    version-8 tables` warnings, a burst of pthread creation, then silence.
    (`crates/raeen-hle/src/libsce_agc.rs:4832` accepts only versions 8/10/12.)

**Not established:** whether any of the four ever called `sceVideoOutOpen`. The
logs cannot answer it, for the reason in §1. The frame-path summary is what
decides it, and the run in §4 is what produces it.

An earlier draft of this document claimed the four "never reach VideoOut,"
inferred from zero `sceVideoOut` grep hits. That inference was wrong — Minecraft
fails the same grep — and it is recorded here so the mistake is not repeated.

---

## 3. KytyPS5 differential: we acknowledge vblank, and never deliver it

KytyPS5 (`reference/kytyps5`, MIT © InoriRus / Nmzik) boots Blasphemous II fully
in-game. Diffing its presentation subsystem against ours surfaces one
structural difference that exactly matches the observed symptom — a title parked
at near-zero CPU, forever, with nothing logged.

**KytyPS5 has a free-running, guest-independent vblank source. Raeen has none.**

KytyPS5's host window loop ticks vblank every displayed frame regardless of what
the guest is doing:

* `src/graphics/presentation/window/window.cpp:350-354` — `GameShowWindow` calls
  `VideoOutBeginVblank()` → `VideoOutFlipWindow(0)` → `VideoOutEndVblank()`.
* `src/graphics/presentation/videoOut.cpp:649-686` — `VideoOutContext::VblankBegin`
  / `VblankEnd` increment the pre-vblank and vblank counters and call
  `TriggerVideoOutEventsLocked(...)` for **every opened handle**, driven purely
  by the host clock.
* `src/graphics/presentation/videoOut.cpp:402` — `WaitForNextVblank()` paces it
  against `Config::GetVblankFrequency()`.

Raeen's vblank sequence advances from exactly two places, **both guest-driven**:

* `crates/raeen-hle/src/libsce_video_out.rs:357` — inside `hle_submit_flip`
  ("a completed flip implies a display refresh").
* `crates/raeen-hle/src/libsce_video_out.rs:922` — inside `hle_wait_vblank`.

Our own source comment states it plainly, at `hle_add_vblank_event`:

> Raeen instead ticks vblank events from `sceVideoOutWaitVblank` and from every
> completed flip, which keeps event-driven frame loops advancing without a host
> timer thread.

That holds for a *polling* frame loop. It fails for an **event-driven** one.
A guest thread that

1. opens video out,
2. registers a vblank event on an equeue (`sceVideoOutAddVblankEvent` → we
   return `SCE_OK` and store the registration), then
3. blocks in `sceKernelWaitEqueue` waiting for the first vblank **before**
   submitting its first flip,

waits forever. Nothing in Raeen will ever fire that event: the only two tickers
are the two calls the guest is not going to make, because it is blocked. Zero
CPU, zero log output, until the harness kills it at 180 s. Under KytyPS5 the
host loop delivers the event within ~16 ms and the title proceeds.

This is the same bug class the project hit twice earlier the same day —
`sceKernelRaiseException` acknowledged but never delivered, and `HLE_ERROR`
leaking as a guest-visible return. **We ack; we never deliver.**

### Related VideoOut event gaps

KytyPS5 implements six VideoOut event entry points; Raeen implements two.

| Export | KytyPS5 | Raeen |
|---|---|---|
| `AddFlipEvent` | `videoOut.cpp:1064` | yes |
| `AddVblankEvent` | `videoOut.cpp:1079` | yes |
| `DeleteFlipEvent` | `videoOut.cpp:1059` | **missing** |
| `DeleteVblankEvent` | `videoOut.cpp:1069` | **missing** |
| `AddPreVblankStartEvent` | `videoOut.cpp:1085` | **missing** |
| `DeletePreVblankStartEvent` | `videoOut.cpp:1074` | **missing** |
| `AddOutputModeEvent` | `videoOut.cpp:1091` | **missing** |

Blasphemous II imports `sceVideoOutDeleteFlipEvent` (NID
`0xfcece7d05d401518`), and it is **unresolved** in our build — confirmed in its
own log:

```
missing sceVideoOutDeleteFlipEvent — NID 0xfcece7d05d401518 (-Ozn0F1AFRg)
  wanted from library 'libSceVideoOut'
```

62 of its imports are unresolved in total, including three `libSceAgc` NIDs
(`0x93413bbe482a02e1`, `0x81092a90bb6d729c`, `0x2247ddb7fac8a821`) and
`sceKernelSleep`. Separately: the compat report records
`unresolved_nids: []` for this run, so the report builder is not picking up the
loader's `missing ...` lines — a second tooling gap, out of scope here.

---

## 4. The run that settles it

One 60-second Blasphemous II launch with both observers armed:

```powershell
cargo build --release -p raeen-gui

$env:RAEEN_FRAME_PATH  = "5"     # frame-path summary every 5 s
$env:RAEEN_STALL_DUMP  = "1"     # per-thread HLE ring + guest RIPs + host backtraces
$env:RAEEN_TIME_HLE    = "1"     # per-thread wall-clock sinks (names the blocking call)
$env:RAEEN_TRACE_EINVAL = "1"    # populates the HLE call ring
$env:RAEEN_LOG         = "info,raeen_hle::libsce_video_out=debug"

.\target\release\raeen.exe --run-eboot "E:\PS5\PPSA13580-app\eboot.bin" `
  2>&1 | Tee-Object -FilePath blasphemous-framepath.log
# Ctrl-C after ~60 s
```

Then read, in this order:

1. `frame path: reached=` — the last rung reached. `reached=nothing` means the
   guest never opened a video-out handle and the stall is upstream of graphics
   entirely; anything else names the exact next stage to investigate.
2. The stall dump's `inflight` list — the HLE call each thread is *currently
   inside*. An empty entry means that thread is blocked in guest code or
   runtime infrastructure, not in an HLE call.
3. The stall dump's per-thread `top` sinks — a thread whose single top entry
   accounts for the whole run is parked in that one call. **If that call is
   `sceKernelWaitEqueue` on a thread that registered a vblank event, §3 is
   confirmed** and the fix is the design in §5.
4. The guest RIPs (`t<N>@module+0xoffset`) — for Until Dawn, the spinning
   title, this is the primary evidence; decode with `--dump-vaddr`.

Expected cost: the stall dump suspends every guest thread every 6 s, so do not
use this configuration for a performance measurement.

---

## 5. Design: a host vblank source

> **IMPLEMENTED 2026-07-28, default OFF** — `crates/raeen-hle/src/host_vblank.rs`,
> gated behind `RAEEN_HOST_VBLANK`. What was built, the equeue-wake seam that
> removed the blocker below, the single-owner rule, and the retail A/B that
> decides whether it ships on by default: **`docs/host-vblank-source.md`**.
> Still unmeasured against a title; §4's run and that A/B are what settle it.

Deliberately **not** implemented in the change that wrote this document. It is
not a small fix, and shipping an unverified pacing change would risk Minecraft
and ASTRO.BOT, which currently render.

Why it is not small: `HleContext` is a borrowed struct of `&dyn` trait objects
with a lifetime (`crates/raeen-hle/src/lib.rs:653`), and both
`trigger_vblank_events` and `kernel_equeue::wake_equeue` take it. A background
host thread cannot hold one. Delivering vblank off the guest's back therefore
needs a host-thread-safe path to the process's equeues — an `Arc<OrbisKernel>`
plus a wake routine that does not require the full `HleContext`. That is a
refactor of the equeue wake seam, not a patch.

Sketch, following KytyPS5's structure:

1. Extract the equeue wake path so it can be driven from an
   `Arc<OrbisKernel>` alone (no `mem`/`alloc`/`gpu` borrows). This is the real
   work and the only risky part.
2. Add a process-scoped vblank ticker thread started on the first successful
   `sceVideoOutOpen`, paced by the existing `configured_vblank_period()`
   (`RAEEN_VBLANK_HZ`, default 60 Hz), that performs KytyPS5's
   `VblankBegin`/`VblankEnd` equivalent: advance `video_out_vblank_count` and
   trigger vblank events for every open handle.
3. Make `hle_wait_vblank` and `hle_submit_flip` *observe* that sequence rather
   than being the ones to advance it, so vblank cannot be double-counted.
4. Gate behind `RAEEN_HOST_VBLANK=1` for the first hardware A/B, then flip the
   default once Minecraft and ASTRO.BOT are re-measured unchanged.

All four steps are done as described, with two documented deviations (the ticker
starts at process bootstrap rather than on the first `sceVideoOutOpen`, because
`hle_open` has only a borrowed kernel — and free-running from display init is
KytyPS5's behavior anyway; and step 4's default stays off pending the A/B).

Also worth doing independently, and cheap: implement the five missing VideoOut
event exports from the §3 table. **Done** — all five are registered in
`libsce_video_out.rs`, so the §3 table is stale in Raeen's favor.

**Do not treat §3 as proven for this cluster** until step 3 of §4 shows a
thread parked in `sceKernelWaitEqueue`. It is a real defect either way — a
title that waits on vblank before its first flip deadlocks on Raeen today — but
whether it is *this* cluster's first blocker is exactly what the run in §4
measures.

---

## Attribution

`reference/kytyps5` (MIT, © InoriRus / Nmzik) was read for design comparison
only. No code was copied into this tree by this change; §3 and §5 cite file and
line so the comparison can be re-derived. If the §5 design is implemented from
that structure, record it in `THIRD_PARTY_NOTICES.md` and
`docs/reference-port-ledger.md` at that point.
