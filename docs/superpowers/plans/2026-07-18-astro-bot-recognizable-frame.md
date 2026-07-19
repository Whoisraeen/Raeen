# ASTRO.BOT Boot-to-Recognizable-Frame Plan

**Date:** 2026-07-18  
**Target:** the user's decryptable ASTRO.BOT installation, supplied out of tree  
**Milestone:** M5 only after a recognizable 3D frame is captured and the title remains responsive

## Goal

Make the release build boot ASTRO.BOT through its normal process entry, reach the
AGC/PM4 pipeline, execute the scene's draw and compute work, feed compute-written
HDR resources into the final composite, and present changing, recognizable 3D
frames. The normal launch must not require proprietary firmware, Sony keys, or an
environment-variable workaround.

"Running" means all of the following:

1. The title reaches the graphics workload repeatably in a release build.
2. Scene-critical shaders translate and execute; skipped work is named and bounded.
3. Compute outputs are visible to later draw/composite passes.
4. VideoOut flips at least two distinct, non-flat frame images.
5. A recognizable 3D frame is captured, the process survives a five-minute run,
   and pad input produces an observable title or frame-state change.

This is not a promise of full-title compatibility. Menus, audio, saves, networking,
and later gameplay can remain incomplete if they do not block the acceptance run,
but every remaining limitation must be documented.

## Measured starting point

Two different states must not be confused:

- A prior 60-second run reached 541 draws, 785 compute dispatches, 160 submissions,
  and 18 flips, but every captured frame was the same flat green composite.
- The current release regressed before graphics. It executes the main module's
  `DT_INIT`, then the title CRT executes that initializer again. A constructor
  inserts the same node twice, and `_start` loops over the resulting cyclic list at
  `module+0x7426c00`.
- The existing `XPS5X_SKIP_MAIN_INIT=1` diagnostic proves the cause: with loader-side
  main initialization skipped, the cyclic-list loop disappears and the title moves
  to the next blocker in roughly five seconds.
- The next blocker is the POSIX `clock_gettime` NID
  `0x94b313f6f240724d`, imported from provider library `libkernel`. The behavior is
  already implemented as a thin adapter in `libScePosix`; the measured provider
  binding does not currently resolve it.
- On the graphics side, storage buffers execute and copy back, but compute storage
  images are rejected by `prepare_stage_binding`. The compute executor is one-shot
  and does not share image objects with later draw or VideoOut consumers. This is
  the known reason the final composite can continue sampling empty HDR resources.

The first two fixes are therefore boot correctness, not speculative GPU work.

## Architectural constraints

- Keep one runtime. Diagnostic mode records the normal runtime; it does not replace
  scheduling, HLE, or GPU execution with a second implementation.
- HLE exports remain ABI adapters. Time comes from `TimeSubsystem`, waits from
  `WaitSubsystem`, guest memory through validated capability types, and GPU work
  through the XPS5X-owned submission interface.
- `GuestProcess` owns module initialization state, address space, threads, TLS,
  handles, HLE state, and its GPU session. New title-specific global state is not
  acceptable.
- Kyty shader/PM4/Vulkan mechanisms stay behind `xps5x-gpu` contracts. No Kyty type
  becomes part of a public runtime or HLE API.
- HLE is the default per-module policy. LLE is used only for useful, user-supplied,
  decryptable modules. Normal operation must not depend on proprietary firmware.
- Do not add title binaries, dumps, keys, shader captures, or screenshots to git.
  Automated tests use synthetic fixtures derived from behavior, not title bytes.

## Delivery order

### Slice 1 - Make process initialization execute exactly once

**Files:**

- `crates/xps5x-firmware/src/dynlib/linker.rs`
- `crates/xps5x-firmware/src/lib.rs`
- `crates/xps5x-runtime/src/lib.rs`
- `crates/xps5x-runtime/tests/execute.rs`

**Implementation:**

1. Replace the implicit "last initializer is main" convention with an explicit
   `ModuleInitRole::{Dependency, Main}` on `ModuleInit`.
2. Add an explicit entry policy to runtime execution:
   `CrtOwnsMainInit` for `execute_process` and `LoaderOwnsMainInit` for direct
   function/module execution that does not enter a CRT.
3. In process mode, run dependency initializers in dependency order, then enter
   `_start`; do not pre-call the main initializer. Keep dynamically loaded PRX
   initialization under `sceKernelLoadStartModule` ownership.
4. Remove the environment variable from normal correctness. It may remain for one
   release as a deprecated diagnostic assertion, but the default must take the
   proven path.
5. Record each initializer transition with the process id, module name, role, and
   stable diagnostic sequence number.

**Tests first:**

- A synthetic dependency initializer and main CRT initializer each increment a
  distinct guest counter. `execute_process` must observe dependency = 1, main = 1.
- A synthetic main whose initializer inserts a node must not be invoked twice.
- Direct module/function execution must retain its explicit loader-owned behavior.
- Initialization order must be deterministic across repeated runs.

**Exit gate:** a default release launch, with no `XPS5X_SKIP_MAIN_INIT`, does not
visit the `0x7426c00` cycle and reaches the next guest import.

### Slice 2 - Resolve `clock_gettime` through the time interface

**Files:**

- `crates/xps5x-hle/src/libsce_posix.rs`
- `crates/xps5x-hle/src/libkernel.rs`
- `crates/xps5x-hle/src/lib.rs`
- `crates/xps5x-firmware/src/dynlib/linker.rs`
- `crates/xps5x-firmware/tests/hle_nid_coverage.rs`

**Implementation:**

1. Add a regression test using the measured provider-library identity and NID,
   not only a global name-to-NID lookup.
2. Inspect the parsed provider reference. If the import genuinely names
   `libkernel`, register `libkernel::clock_gettime` as a thin POSIX ABI adapter to
   the existing implementation. If the parser incorrectly collapsed the module
   and library identities, fix that mapping instead. Do not add cross-library
   fallback resolution, which would make provider policy nondeterministic.
3. Keep time behavior in `TimeSubsystem`: realtime uses `wall_clock`, monotonic
   clocks use `monotonic_elapsed`, and sleep remains behind the same process-owned
   services.
4. Validate the guest `timespec` through a writable guest range and return the
   POSIX convention. HLE must not dereference a raw guest address directly.

**Tests first:**

- The measured `(provider, NID)` resolves to an implemented HLE trampoline.
- Realtime and monotonic `timespec` layouts are valid and nanoseconds stay below
  one billion.
- Invalid output pointers return the correct error without host memory access.
- Diagnostic mode records call, arguments, return, and stable sequence ordering.

**Exit gate:** the title passes `clock_gettime`; the next blocker is reported by
name/NID or the first AGC submission is observed.

### Slice 3 - Drive boot blockers to AGC with an evidence loop

Do not try to implement all 420 currently unresolved relocations. Most may never be
called. Run a bounded release diagnostic after each fix and implement only the
first import, exception, wait, or ownership violation that blocks forward progress.

For every newly hit HLE export:

1. Capture provider module, provider library, NID, arguments, return address, guest
   thread, and the last 64 deterministic diagnostic events.
2. Put behavior in the owning subsystem. The HLE function only validates ABI inputs,
   calls the interface, and translates the result.
3. Add a provider-aware linker test and an ABI/guest-memory test before rerunning.
4. Prefer a loud unsupported result to a success stub that corrupts state.

Expected areas are time, waits/events, VFS, and thread ownership. Pthread mutexes
are not the first root cause: the previous `last_hle=MutexUnlock` value was merely
the last historical call before the cyclic list. Rework synchronization only if a
new deterministic trace proves it is the active blocker.

**Exit gate:** a 60-second release run reaches AGC submissions without an
unimplemented-import fault, dead wait, or initializer cycle.

### Slice 4 - Make diagnostics produce a comparable run summary

**Files:**

- `crates/xps5x-core/src/diagnostics.rs`
- `crates/xps5x-runtime/src/dispatch.rs`
- `crates/xps5x-gpu/src/agc_exec.rs`
- `crates/xps5x-gui/src/main.rs`

Extend the existing deterministic recorder, still around the same runtime, with a
generic `--diagnostic-dir <path>` output and a final `run-summary.json` containing:

- process/module initialization transitions;
- HLE enter/exit pairs and first unresolved import;
- waits, wake reasons, event transitions, and task/thread ownership;
- AGC submissions, draws, compute dispatches, skipped shaders grouped by stable
  reason, and VideoOut flips;
- frame hashes, distinct-frame count, simple entropy/range checks, and the final
  watchdog snapshot.

Counters and records receive stable sequence numbers at the event source. Paths and
host timestamps are metadata, not ordering keys. The recorder stays bounded and
dumps its retained tail on fault or watchdog termination.

**Exit gate:** two equivalent fixture runs produce the same ordered event kinds and
ownership transitions, and a title run can be compared to the prior
541-draw/785-dispatch/18-flip baseline without reading megabytes of text logs.

### Slice 5 - Restore the scene shader set

**Files:**

- `crates/kyty-graphics/src/shader_*`
- `crates/xps5x-gpu/src/draw_translate.rs`
- `crates/xps5x-gpu/src/agc_exec.rs`

1. Rebaseline the title after boot is restored. Group skips into unbound shader,
   parse failure, SPIR-V generation failure, validation failure, and unsupported
   resource rather than treating them as one count.
2. Close only opcodes and semantics present in scene-critical captures. Each change
   must pass parse -> SPIR-V assembly -> Naga validation tests.
3. Resolve the remaining `EXP target 0x0d/POS1` through formal vertex-output
   semantic metadata. Do not drop components or invent a target mapping merely to
   make validation pass.
4. Eliminate the remaining unbound compute cases by making shader ownership part of
   queue/session state. Cross-queue recovery may diagnose missing ownership, but it
   must not silently bind an unrelated shader.
5. Keep raw captured shader bytes local. Commit minimal synthetic instruction
   fixtures for every supported opcode or semantic.

**Exit gate:** in the same bounded run used for the baseline, the emulator reaches
at least the previous order of work (541 draws, 785 compute dispatches, 18 flips),
and no scene-critical shader is skipped for an unknown opcode or missing binding.

### Slice 6 - Add compute-image coherence and feed the final composite

**Files:**

- `crates/xps5x-core/src/subsystems.rs`
- `crates/xps5x-gpu/src/draw_translate.rs`
- `crates/xps5x-gpu/src/agc_exec.rs`
- `crates/xps5x-gpu/src/vulkan/compute.rs`
- `crates/xps5x-gpu/src/vulkan/offscreen.rs`

This is the main flat-green fix.

1. Extend the XPS5X-owned shader/resource contract with storage-image descriptors.
   The public contract uses guest GPU-visible capabilities, format, dimensions,
   mip/layer range, and access mode; it does not expose Kyty or Vulkan ownership
   types to HLE/runtime callers.
2. Put a process-owned GPU resource table in the GPU session, keyed by validated
   guest GPU-visible range plus descriptor identity. Draw, compute, and VideoOut
   must resolve the same guest resource to the same coherent backing image.
3. Implement the exact storage-image formats and addressing modes observed in the
   title's scene/HDR passes. Unsupported formats remain explicit diagnostic skips.
4. Add Vulkan layout/access transitions and compute-to-fragment barriers. A compute
   write must be visible to the sampled image used by the later composite.
5. Correctness comes before speed: a temporary readback/upload bridge is acceptable
   to prove data flow. Then replace it with persistent images, cached descriptor
   layouts/pipelines, and batched submissions; creating and waiting on a complete
   Vulkan pipeline for each of 785 dispatches is not a viable steady-state path.
6. Keep storage-buffer copyback for CPU-visible buffers, but do not treat it as an
   image-coherence mechanism.

**Tests first:**

- A compute fixture writes a storage image; a later fragment shader samples it and
  produces the expected pixels.
- A multi-pass fixture proves write -> barrier -> sample ordering.
- Aliased descriptors for the same validated GPU-visible range share content while
  incompatible ranges/formats fail explicitly.
- Process teardown releases cached pipelines, images, and descriptor resources.

**Exit gate:** the final composite consumes non-zero, changing compute output; frame
hashes are no longer all identical and the flat-green image is gone.

### Slice 7 - Presentation, interaction, and M5 evidence

1. Verify `sceVideoOutSubmitFlip` presents the registered scanout buffer, not the
   last arbitrary render target.
2. Capture frame hashes before PNG conversion and require at least two distinct
   hashes plus non-flat channel range/entropy.
3. Route pad input through the process-owned input session and show one observable
   response. Audio may remain a non-blocking stub, but it may not stall the title.
4. Run for five minutes under bounded diagnostics. Check guest-thread progress,
   task ownership, waits/wakes, GPU memory growth, submissions, and flips.
5. Capture a recognizable frame locally and document remaining visual or gameplay
   defects. Update the M5 ledger only after this evidence exists.

## Verification commands

Run scoped checks after each slice, then the dependent set:

```powershell
cargo test -p xps5x-runtime
cargo test -p xps5x-firmware
cargo test -p xps5x-hle
cargo test -p kyty-graphics
cargo test -p xps5x-gpu
cargo clippy -p xps5x-runtime -p xps5x-firmware -p xps5x-hle -p kyty-graphics -p xps5x-gpu -- -D warnings
cargo fmt --all -- --check
cargo build --release -p xps5x-gui
```

The local title run must use a user-supplied path and an ignored artifact directory:

```powershell
target\release\xps5x.exe --run-eboot <user-eboot> --diagnostic-dir <ignored-run-dir>
```

Before claiming M5, also run the workspace tests or explicitly record why any
unrelated failure is pre-existing:

```powershell
cargo test --workspace
```

## Acceptance ladder

| Gate | Required proof |
|---|---|
| B0 - Boot restored | Main initializer runs once, dependencies run once in order, no `0x7426c00` loop, and `clock_gettime` resolves through the measured provider. |
| B1 - GPU workload restored | A bounded release run reaches AGC and recovers the prior draw/dispatch/flip order of magnitude with named, bounded skips. |
| B2 - Scene data connected | Compute storage-image output reaches the composite; at least two non-flat frame hashes differ. |
| B3 - Game running | A recognizable 3D frame is captured, the title remains alive for five minutes, and pad input causes an observable response. |
| M5 evidence accepted | B3 proof is recorded, tests are green, known issues are documented, and no proprietary material is committed. |

## Stop conditions and risks

- If a launch needs encrypted retail content that the user has not supplied in a
  decryptable form, stop and report the permanent clean-room boundary; do not add
  keys or firmware dependencies.
- If a shader semantic is not established by the instruction stream, descriptor
  metadata, an allowed reference, or a focused experiment, leave it unsupported
  and record the evidence needed. Do not guess until a frame happens to appear.
- If the GPU path becomes correct but too slow, profile after B2. Pipeline caching
  and batching are performance work; they must not obscure resource-coherence bugs.
- Do not call M5 on a green clear/loading frame. The acceptance image must contain
  recognizable title 3D content.

## Immediate next action

Implement Slice 1 with the double-initializer fixture. The already-proven skip-main
diagnostic becomes the expected default process policy, after which Slice 2 should
be small: make the measured `libkernel` provider resolve the existing thin
`clock_gettime` adapter and rerun to discover the next real blocker.
