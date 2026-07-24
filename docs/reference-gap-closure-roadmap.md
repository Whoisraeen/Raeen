# Raeen Reference Gap-Closure Roadmap

This roadmap measures Raeen against KytyPS5 and SharpEmu without turning
reference function counts or synthetic fixtures into compatibility claims.
Raeen remains GPL-2.0-only; ports must follow `clean-room` and update
`THIRD_PARTY_NOTICES.md`.

## Current baseline

- M2 and M3 are **open**. Their current tests prove useful AGC/Vulkan, pad,
  CPU-framebuffer and flip infrastructure, but use synthesized guests.
- Retail evidence is reported per measured title run: HLE calls, guest faults,
  submissions, draws, dispatches, flips, presented frames and elapsed time.
- A registration count is an inventory metric, not compatibility. A function
  returning a fabricated success is not equivalent to a working implementation.

## Implemented 2026-07-23

- The production Shell now starts guest execution in a child `raeen
  --run-eboot` process assigned to a Windows Job Object with
  `KILL_ON_JOB_CLOSE`. A deliberate hard-abort test proves the Shell-side
  process survives. Native XInput/DualSense polling runs inside the child.
- Reviewed libc leaf imports (`strlen`, `strcmp`, `strncmp`, `memcmp`, `bcmp`)
  use executable eight-byte slots and a shared SysV gateway. The gateway
  switches to a private 256 KiB host stack and forwards six register
  arguments, eight stack arguments and XMM0â€“XMM7. Exit, fiber, native-trap,
  callback-capable and unclassified calls retain VEH.
- Measured one-million-call benchmark:
  - VEH: 4.024 s, 248,480 calls/s, 1,000,000 HLE access violations.
  - executable gateway: 0.626 s, 1,598,025 calls/s, zero HLE access violations.
  - measured dispatch speedup: 6.4Ã—.

Run the benchmark from the repository root:

```powershell
$env:RAEEN_DISABLE_DIRECT_HLE='1'
cargo test -p raeen-runtime --test execute benchmark_one_million_executable_hle_calls -- --ignored --nocapture
Remove-Item Env:RAEEN_DISABLE_DIRECT_HLE
cargo test -p raeen-runtime --test execute benchmark_one_million_executable_hle_calls -- --ignored --nocapture
```

These are infrastructure wins, not M2/M3 closure evidence.

## P0 — correctness and containment

1. Split `raeen.exe` into Shell and `raeen-runner` processes.
   - Shell owns library, settings, themes and crash UI.
   - Runner owns guest VA, VEH, HLE, Vulkan, audio and input forwarding.
   - Windows runner lives in a Job Object with kill-on-close.
   - IPC carries launch configuration, structured logs, compatibility counters,
     input snapshots, frame handles/copies and a final crash report.
   - Acceptance: deliberate guest AV kills only the runner; Shell remains usable
     and publishes the exact fault/module/recent-HLE report.
2. Keep user game data, keys and firmware outside the repository and outside
   compatibility artifacts.

## P1 — zero-fault HLE imports

Replace the permanent `PAGE_NOACCESS` resolved-import region with executable
per-import thunks. Retain fault sentinels only for unresolved imports and
control-transfer cases that genuinely need a captured machine context.

The bridge must:

- preserve SysV guest GPRs, `AL`, XMM0–XMM7 and arguments 7+;
- switch from the guest stack to a host stack before entering Rust;
- translate SysV guest arguments to the Windows x64 host ABI;
- preserve/re-arm the guest FS base and maintain per-guest-thread context;
- support blocking HLE calls without holding the diagnostic guest GIL;
- route exit, fiber switching, exception/unwind and native traps through the
  existing context-changing slow path;
- emit unwind metadata or prohibit unwinding across generated code;
- use W^X pages: write, flush instruction cache, then execute-read.

Stages:

1. Add a deterministic benchmark for one million leaf HLE calls and record VEH
   exception count, wall time and calls/second.
2. Add an executable thunk for a classified pure leaf function.
3. Prove return value, six register arguments, eight stack arguments, float
   arguments, callee-saved registers, TLS and nested guest callbacks.
4. Expand classification to ordinary HLE calls.
5. Acceptance: resolved leaf imports generate zero access violations and show a
   material measured speedup; unresolved and context-changing paths retain their
   diagnostics.

SharpEmu's import trampoline is a useful GPL-2.0 reference, but it is hundreds
of bytes of ABI/state handling rather than a safe `mov rax, imm64; jmp rax`
drop-in. Port the behavior with tests, not the superficial two-instruction
shape.

## P2 — HLE integrity and measured breadth

- CI must run the explicit-NID integrity test. Ordinary known names use
  name-derived registration; only reviewed provider-private/unknown identities
  use `register_nid`.
- Generate per-title unresolved/called/error/hot-call reports from nightly runs.
- Prioritize functions by the earliest measured title blocker and call
  frequency, not raw catalogue count.
- Add a SharpEmu-style build-fail duplicate/provider/NID integrity report.

## P3 — retail GPU path

1. Replace the instruction-at-a-time emitter with a real shader IR:
   CFG, SSA values, dominance, PHI construction, EXEC/VCC/SCC modeling,
   structured/relooped control flow and typed resource operations.
2. Finish FLAT/global/scratch/LDS semantics and invalidate shader/resource
   caches when tracked guest CPU writes overlap their inputs.
3. Move detiling/deswizzle to compute shaders; keep CPU fallback only for
   diagnostics and tiny transfers.
4. Implement persistent color/depth targets, alias tracking, barriers,
   descriptor reuse, pipeline caches and batched command submission.
5. Implement a real Vulkan swapchain/present path with resize, HDR/SDR,
   fullscreen, VRR and frame pacing.
6. Acceptance: a named legal title produces recognizable presented frames.
   Skipped draws and offscreen PPM fixtures do not close M2–M5.

## P4 — media and controller output

- Implement AJM job parsing and legal host codec backends for ATRAC9/AAC/MP3;
  preserve a deterministic silence fallback with explicit compatibility status.
- Add DualSense HID output for rumble, lightbar and adaptive triggers behind a
  device-capability layer. Never let output failures block input.
- Acceptance scenarios cover sustained PCM, movie playback, reconnect,
  controller output and clean shutdown without leaked threads/devices.

## P5 — portability and license automation

- Abstract guest address-space reservation, fault handling, TLS/FS switching,
  executable thunks, host audio, input and presentation.
- Linux: `mmap`/signals/pthreads/Vulkan first. macOS follows only with an
  explicit x86-64/Rosetta and MoltenVK support contract.
- Add SPDX headers, `LICENSES/`, dependency-license policy and REUSE linting.
  Per-file machine checks complement—not replace—`THIRD_PARTY_NOTICES.md` and
  clean-room review.

## Nightly scorecard

For Astro Bot, Minecraft, one UE5 title and one smaller title, publish:

- boot stage and elapsed time;
- resolved/unresolved/called HLE functions;
- HLE calls/second and exception count;
- guest/host crashes and leaked runner status;
- shaders translated/refused/cache-hit/invalidated;
- submissions/draws/dispatches/flips/presented frames;
- real frame time, generated frame time (if any), CPU and GPU time;
- peak working set, committed guest memory and persistent-cache size;
- audio underruns and controller input/output acceptance.

Compatibility is published only from these artifacts.
