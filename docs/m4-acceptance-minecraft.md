# M4 acceptance — Minecraft (Bedrock, retail PS5)

**Gate (from CLAUDE.md):** *One legal commercial 2D title to interactive menu;
crash/log shows syscalls/NIDs/GPU faults; save-data host map.*

**Status: MET**, on the evidence below. Everything here is an iron run against
a **user-owned retail disc/PSN title** — no Sony keys, SDK, or firmware in the
repository (see `docs/clean-room.md` boundaries). The title is not
redistributed and is not in-tree; reproduce with your own copy.

Recorded 2026-07-27. Host: Ryzen 5 7640HS (12 threads), Radeon 760M iGPU,
Windows 11 Pro, Vulkan 1.3.

---

## Reproduce

1. Install the title so `<game folder>/eboot.bin` exists, and add the parent
   directory in **Settings ▸ Game Folders** (or drop it in a watched folder —
   the library auto-rescans).
2. Launch it from the Home rail.
3. Wait ~45–60 s for the main menu (first run is slower: shader translation
   populates the on-disk cache).

## What must be observed

| # | Gate clause | Observation |
|---|---|---|
| 1 | Commercial title boots | Title renders its own menu: 3D panorama, logo, `Play` / `Settings` / `Store`, `Sign in to PlayStation™Network`. |
| 2 | **Interactive** menu | Guest input moves guest focus. D-pad Down moves the highlight `Play` → `Store`; Up returns it. Cross on `Play` opens the Worlds screen (`Worlds 1`, `Friends 0`, `Create New`, `Realms`, `Storage`). |
| 3 | Save-data host map | The Worlds list shows the **persisted** world (`My World` / `Survival`) with its thumbnail, and it loads. Log: `VFS: /savedata0/ -> savedata\PPS<id>-app\BedrockWorld<...>`. Saves land under `savedata/<title>/` in the repo working directory. |
| 4 | Logs are actionable | Verified by outcome, not by existence — see below. |
| 5 | Beyond the gate | The title reaches **in-world gameplay**: rendered 3D terrain with textures, HUD (hearts, hunger, hotbar), and button prompts, running at **56 FPS**. |

## Clause 4: the logs are actionable, demonstrated

Two blockers in this run were found *from the logs alone* and fixed:

1. **Sync starvation.** `scePthreadMutexLock stuck >3s — deadlock; naming the
   holder mutex=… waiter=3 waiter_name="Streaming Pool(1)" owner=5
   owner_name="Streaming Pool(3)" ty=3 recursion=1` named the exact mutex, the
   waiter, and the holder. That pointed straight at contended waits spinning
   on `yield_now()` and starving the owner. Fixed by parking waiters on a host
   condvar (commit `eef31c1`); in-game CPU fell from all-cores-busy to **0.42
   of 12 cores** and the deadlock reports went to zero.
2. **Untextured world.** `guest shader analysis failed — draws binding it will
   be skipped stage="ps" addr=0x1700ab00 reason=… shader analysis not
   implemented: ps: direct sgprs` named the stage, the guest address, and the
   precise unimplemented feature. Fixed by allowing direct user SGPRs in Gen5
   pixel shaders (commit `d21e727`), which is what took the scene from flat
   colour blocks to real block textures and 4 → 56 FPS.

Also available and exercised: `Settings ▸ Advanced` toggles (HLE call tracing,
GPU-resource/shader/frame dumps, call stats) bridged per-launch to the runner
process; out-of-process minidumps into `logs/crashes/`; `RAEEN_PROFILE=1` for
a live puffin frame timeline.

## Honest limitations

- **Not a performance claim.** 56 FPS was measured at one location in one
  world; Minecraft Bedrock is a light forward-rendered title and is not
  representative of PS5 3D workloads on this iGPU.
- **A long-session hang was seen before the shader fix** (main thread parked on
  a mutex, frames frozen). It did not reproduce in the 56 FPS run, but the
  in-world path has not been soak-tested — treat stability as unproven.
- **Rendering is not verified correct**, only recognisable: unknown PM4
  context registers are skipped (mostly per-MRT colour-buffer descriptors and
  DCC/CMASK/FMASK compression metadata), and MRT1–7 binding plus fast-clear
  words are unimplemented. Deferred-rendering titles will need those.
- **Menu navigation was driven by synthetic input** (`PostMessage` to the
  Shell window, merged into the guest pad state) — the same path a real pad
  takes, but a physical DualSense was not in the loop for this record.
- This does **not** claim M5. M5 wants a named 3D title with recognisable
  frames *and* documented known issues for that title; Minecraft's in-world
  rendering is promising evidence toward it, not the gate.
