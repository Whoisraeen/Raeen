# Emulator subsystem and process boundaries

This document defines the incremental architecture used by the live native
execution path. It is not a second runtime and does not replace the proven
identity-mapped `GuestArena` + VEH design.

## Ownership

`raeen-runtime::GuestProcess` is the process lifetime boundary. It owns or
retains ownership handles for:

- the composed main module and file-backed dependencies;
- the identity-mapped address space and executable trampoline guard;
- the HLE registry and process-local `OrbisKernel` state;
- native guest threads, their stacks, TCBs, and static TLS layout;
- kernel handles and loaded-module state;
- one `GpuProcessSession`, including PM4 register state, shader cache,
  framebuffers, and its ordered submission worker.

Process teardown reaps native guest workers, drains/stops GPU submission, and
only then permits guest memory to be unmapped. The Shell receives an observer
clone of the active GPU session so presentation does not own emulator state.

## HLE subsystem contracts

Contracts live in `raeen-core::subsystems`; concrete state remains in the
kernel/GPU crates. `HleContext` exposes the contracts alongside the legacy
kernel reference while adapters are migrated incrementally.

| Contract | Active consumers |
|---|---|
| `TimeSubsystem` | clocks, nanosleep/usleep, offline socket select |
| `WaitSubsystem` | event-flag blocking and wakeups |
| `EventSubsystem` | event-flag create/delete/set/clear/cancel |
| `VfsSubsystem` | open/read/write/sync/close |
| `NetworkSubsystem` | offline socket creation, validation, pacing |
| `GpuSubmissionSubsystem` | graphics and async-compute AGC submissions |

New HLE functions should validate/decode guest arguments, call one of these
interfaces, encode the Orbis result, and return. New direct access to
`OrbisKernel` fields requires a reason that the relevant contract cannot yet
express; extend the contract before duplicating subsystem logic in HLE.

## Deterministic diagnostics

Set `RAEEN_DETERMINISTIC_DIAGNOSTICS=1` to enable the process-owned bounded
event stream. `RAEEN_DIAGNOSTIC_CAPACITY` changes its retained tail (default
65,536 events). Events contain no host timestamp or host-thread identity.

One monotonically increasing sequence covers:

- HLE entry and return;
- wait begin/end and wake reason;
- event transitions;
- guest task/thread ownership and release;
- graphics/async-compute submission.

Events are retained by `OrbisKernel::diagnostics` and emitted to the
`raeen::deterministic` tracing target with their stable sequence field.

## Guest-memory capabilities

Raw guest integers first become `GuestAddress` and overflow-checked
`GuestRange`. Code that needs a mapped range then requests one of:

- `ValidatedGuestRange` for a declared read/write mode;
- `ExecutableGuestMapping` for native entry/callback targets;
- `GpuVisibleGuestRange` for memory consumed by GPU work.

Only the memory backend constructs these proofs. A raw address cannot be
relabelled as executable or GPU-visible by downstream code.

## Kyty boundary

`kyty-graphics` is an internal mechanism of `raeen-gpu`. HLE/runtime public
contracts use Raeen-owned shader metadata and submission types. Conversion to
Kyty's `ShaderMappedData` happens inside `raeen-gpu`; application lifecycle,
process ownership, and presentation remain in `raeen-*`.

## HLE/LLE policy and clean-room boundary

`ModuleRegistry` selects a policy per provider module:

- `PreferHle` (default): HLE first, then a loaded export;
- `HleOnly`: never execute a file-backed provider;
- `PreferLle`: title-supplied module first, with HLE fallback;
- `LleOnly`: require a loaded export.

LLE candidates must come from files supplied by the user/title and must pass
the existing `KeyProvider` decrypt seam. Raeen ships no firmware, Sony keys,
SDK material, or proprietary modules. A normal installation remains fully
HLE-first and firmware-independent.
