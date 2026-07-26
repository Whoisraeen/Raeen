# Raeen compatibility and performance workflow

This is the production loop for turning owned PS5 software into reproducible,
privacy-safe engineering evidence. It does **not** promise that every title can
run at 120 FPS today. “Max FPS” means uncapped host-side presentation, no
intentional vblank wait, a release build, and measured regressions; game frame
caps, missing emulation, GPU limits, and CPU limits still apply.

## Clean-room and privacy boundary

- Games, firmware, keys, raw logs, shader dumps, screenshots, absolute paths,
  machine details, and generated registries remain local and gitignored.
- Reference projects live only under `reference/`. Do not edit them or merge
  them wholesale. Port one behavior with a failing Raeen test, attribution, and
  a scoped license check.
- Only the sanitized schema from `cargo xtask compat run` may feed the public
  compatibility table. A title is never promoted by opinion or a screenshot
  alone.
- Retail encrypted SELF still requires legal user-provided material. Raeen does
  not ship Sony keys, firmware, SDK content, or proprietary game data.

## One-time setup

Set one or more library roots in the ignored `config.toml`:

```toml
[paths]
game_folders = ["Games", 'E:\PS5']
```

The current machine uses `E:\PS5`; `E:\PS5\Games` does not currently exist.
Discovery searches at most six directory levels, skips symlinks, finds every
`eboot.bin`, extracts `PPSA` IDs where available, hashes content, and removes
duplicate images:

```powershell
cargo xtask compat discover
```

The resulting registry is local at `artifacts/compat/registry.json`. Paths and
aliases are deliberately not publishable.

## Baseline every configured title

Build once, then run every unique executable for a bounded interval:

```powershell
cargo build --release -p raeen-gui
cargo xtask compat run --tier all --profile max-fps --timeout 180
```

The runner launches `raeen.exe --run-eboot`, disables the synthetic vblank
wait, enables draw/call telemetry, captures stdout and stderr without pipe
deadlocks, enforces a timeout, and measures:

- wall time, process CPU time, peak working set, and exit status;
- observed VideoOut flip events;
- shader, GPU, audio, and input event/error counts;
- the first normalized blocker and hashes of the blocker and complete log.

Raw evidence is stored under `artifacts/compat/raw/`. The sanitized run report
is `artifacts/compat/latest.json` and conforms to
`compat/schema-v1.json`. `observed_fps` stays null until Raeen emits a reliable
machine-readable frame-time series; flip counts are not mislabeled as FPS.

Save a local baseline and compare later builds:

```powershell
Copy-Item artifacts\compat\latest.json compat\local\baseline.json
cargo xtask compat compare --baseline compat/local/baseline.json
```

## NID coverage per title

Static, no execution: parse each title's `eboot.bin` plus every on-disk
NEEDED `.prx`/`.sprx` (the same `inspect_module` path the loader uses), then
classify every unique (provider, NID) import exactly like the linker — HLE
registration, LLE via the title's own shipped modules (keyed by file name,
mirroring `load_process`), else unresolved. Render-path libraries
(`libSceAgc*`, `libSceVideoOut*`, `libSceGnm*`, `libSceShader*`) are broken
out separately:

```powershell
cargo xtask nids coverage                 # every registered title
cargo xtask nids coverage --eboot PATH    # one executable + its .prx chain
cargo xtask nids coverage --full          # also list every unresolved import
```

The local report is `artifacts/compat/nid-coverage.json` (gitignored; names
titles/modules, never publish raw). "Resolved" means *a symbol is
registered or shipped* — it is not evidence the implementation is correct
(coverage ≠ rendering). Anonymous NIDs are dictionary-fill targets for
`crates/raeen-firmware/examples/hunt_nid_names.rs`.

## Nightly driver
The nightly tier automatically selects one executable for each role:

- Astro Bot;
- Minecraft;
- the smallest discovered UE5 candidate (currently Until Dawn or Dragon Ball);
- the smallest “small-title” candidate (Subnautica or A Plague Tale).

Run it manually:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run-compat-nightly.ps1
```

Use Windows Task Scheduler to invoke that command nightly from the repository
root. The script builds release, rediscovers the library, drives the four-title
tier, runs audio/input/UI contract tests, and regenerates a sanitized local
table. Do not schedule concurrent runs; Vulkan caches and raw evidence share
local directories.

Install or update the supplied 02:00 task (the time is configurable):

```powershell
powershell -ExecutionPolicy Bypass -File tools\install-compat-nightly-task.ps1
```

## Blocker-to-patch loop

For each title, work only on its first deterministic blocker:

1. Reproduce it twice with the same build and content hash.
2. Reduce it to the smallest unit/integration test possible.
3. Check references in order: KytyPS5, shadPS4, PS5SDK/fixtures, SharpEmu, then
   Kyty shared graphics. Confirm the specific file’s compatible license.
4. Implement the smallest Raeen behavior, with no game-specific address hacks.
5. Run scoped tests and clippy for touched crates.
6. Re-run the same title and compare CPU, RAM, shader errors, flips, stage, and
   first-blocker signature.
7. Run the nightly tier before accepting renderer/runtime changes.

If a blocker survives two focused rounds, run the same title in a reference
emulator and compare the first divergent command/resource state. Never copy
logs containing proprietary paths into committed issues or reports.

## Renderer performance order

Optimize only after correctness evidence. The current order is:

1. complete FLAT/global-memory descriptor and guest-address wiring end to end;
2. reuse persistent color/depth targets by identity and invalidate safely;
3. batch command submission and fences at flip/dependency boundaries;
4. cache translated shaders and Vulkan pipelines persistently;
5. remove synchronous flip/readback and redundant VFS reads;
6. profile CPU hot paths and allocation churn;
7. tune frame pacing and optional presentation caps.

Every optimization requires unchanged-or-better rendered evidence and a
measured CPU/RAM/wall-time comparison. “120+ FPS” can only be claimed for a
specific title, build, scene, machine class, and measurement interval.

## Audio, input, and UI acceptance

Run the automated contract scenarios:

```powershell
cargo xtask acceptance run
```

This tests `raeen-audio`, `raeen-input`, `raeen-hle` audio/pad integration, and
the `raeen-gui` library/presentation contracts, writing a sanitized report.
Before a release, also perform hardware acceptance:

- audio: audible stereo output for five minutes, no hang, bounded underruns;
- input: DualSense/XInput connect, buttons, sticks, triggers, disconnect and
  reconnect;
- UI: library rescan, launch, return to Shell, settings persistence, fullscreen,
  useful crash log, and a screenshot pass using `.agents/skills/verify`.

Automated unit tests do not substitute for those hardware checks.

## Daily upstream delta

Run locally:

```powershell
cargo xtask refs report --fetch
```

The scheduled GitHub workflow clones all four read-only references, compares
their upstream branches with `compat/reference-state.json`, and uploads a
30-day Markdown report. Review deltas; do not auto-merge them. When a port is
accepted, update `THIRD_PARTY_NOTICES.md`, `docs/reference-port-ledger.md`, and
the relevant baseline revision.

## Publishing rule

Generate the table only from a measured sanitized report:

```powershell
cargo xtask compat publish
```

The publisher rejects mismatched schema versions and any result not marked as
measured. The generated `compat/COMPATIBILITY.md` remains ignored until the
project deliberately decides to publish a reviewed snapshot. Stages are narrow:
`rendering` means observed flip evidence, not “playable”; menu/in-game/playable
claims require future explicit visual and input acceptance evidence.
