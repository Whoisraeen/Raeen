# Reference port ledger

**Goal:** port every useful module under `reference/*` into XPS5X Rust crates.  
**Delete rule:** remove `reference/<name>/` only when that reference’s status is
`fully_ported` (all rows `done` or `skip`) and `THIRD_PARTY_NOTICES.md` still
attributes the upstream. Never delete mid-port.

**Status values (row):** `todo` · `wip` · `done` · `skip`  
**Status values (reference):** `active` · `fully_ported` · `deleted`

Update this file in the same session as each port batch. Link commit SHAs.

Claude `/goal` (≤200 chars):

```
/goal Port every useful module in reference/* into XPS5X. Log status in docs/reference-port-ledger.md; delete a ref tree only when its ledger says fully ported.
```

---

## Index

| Reference | Path | License | Upstream | Status | Delete when |
|-----------|------|---------|----------|--------|-------------|
| Kyty | `reference/kyty` | MIT | https://github.com/InoriRus/Kyty | `active` | all rows done/skip |
| SharpEmu | `reference/sharpemu` | GPL-2.0 | https://github.com/par274/sharpemu | `active` | all rows done/skip |
| KytyPS5 | `reference/kytyps5` | (check) | https://github.com/Nmzik/KytyPS5 | `active` (optional clone) | all rows done/skip |
| shadPS4 | `reference/shadps4` | GPL-2.0 | https://github.com/shadps4-emu/shadPS4 | `active` (optional) | patterns only; keep if still consulting |
| PS5SDK | `reference/ps5sdk` | GPL-2.0 | https://github.com/PS5Dev/PS5SDK | fixtures | keep for M1 toolchain builds |

---

## Kyty (`reference/kyty`) — status: `active`

Primary full port. Plan: `docs/superpowers/plans/2026-07-13-kyty-full-port.md`.

### lib/Core → `kyty-core`

| Module | Target | Status | Commit | Notes |
|--------|--------|--------|--------|-------|
| DbgAssert / Common / SafeDelete / Singleton | `kyty-core` | `done` | `4dd3cea` | |
| containers batch (vector, hashmap glue, …) | `kyty-core` | `done` | `daabbaf` | |
| string8 / hashmap / timer / date_time | `kyty-core` | `done` | `c0639a7` | |
| String / Compression | `kyty-core` | `done` | `c81fe71` | |
| JsonReader / Language | `kyty-core` | `done` | `8c1e28f` | |
| Sys (`sys_*.rs`, 9 mods) | `kyty-core` | `done` | (sys-wire batch) | were orphaned (drafted, never declared in lib.rs → never compiled); wired in `#[cfg(windows)]`, lint-fixed, 92 tests now live |
| CharUcd | `kyty-core` | `skip` | | use unicode crate; do not transliterate |
| Database | `kyty-core` | `skip` | | defer rusqlite unless needed |
| VirtualMemory / Threads / File / MemoryAlloc / MSpace | `kyty-core` | `todo` | | after Sys |
| Debug / Subsystems / SDL / Core.cpp | `kyty-core` | `todo` | | port last |

### Later Kyty trees

| Module | Target | Status | Commit | Notes |
|--------|--------|--------|--------|-------|
| lib/Math | `kyty-math` | `todo` | | |
| lib/Scripts | `kyty-scripts` | `todo` | | skip unused lua if not needed |
| emulator/Loader | `kyty-loader` → `xps5x-firmware`/`loader` | `todo` | | |
| emulator/Kernel | `kyty-kernel` → `xps5x-kernel`/`hle` | `todo` | | |
| emulator/Libs | `kyty-libs` → `xps5x-hle` | `todo` | | |
| emulator/Graphics | `kyty-graphics` → `xps5x-gpu` | `todo` | | crown jewel / M2+ |
| emulator top (Audio/Controller/…) | `kyty-emulator` | `todo` | | |

**Delete Kyty when:** every row above is `done` or `skip`, crates wired into hot path or explicitly N/A.

---

## SharpEmu (`reference/sharpemu`) — status: `active`

Second-opinion PS5 emu (C#). Re-implement in Rust; do not vendor C#.

| Area | SharpEmu locus (approx.) | Target | Status | Commit | Notes |
|------|--------------------------|--------|--------|--------|-------|
| LoadStartModule / module handles | KernelRuntimeCompatExports | `xps5x-hle` / firmware | `done` | `fbac0b7` | pseudo-handle approach |
| libc string/mem + atexit | compat exports | `xps5x-hle` | `done` | `c485704` | +review fixes `2ab04fa` (strstr DoS→memchr, truncation warn) |
| libc atoi/strtol/strtoul | compat exports | `xps5x-hle` | `done` | `4f86ea0` | real base-0 parse + endptr |
| time / usleep | kernel time exports | `xps5x-hle` | `done` | `922d0bf` | real host clock; usleep really sleeps |
| GetCompiledSdkVersion / getpid | KernelExports | `xps5x-hle` | `done` | `9457258` | PS5 SDK 9.00 (Gen5); stable pid |
| SELF / eboot / Prospero loader | Loader | `xps5x-firmware` | `todo` | | cross-check vs Kyty |
| PRX / sysmodule load chain | Kernel / Libs | `xps5x-firmware`/`hle` | `todo` | | |
| Fiber / AMPR | Kernel | `xps5x-hle`/`runtime` | `todo` | | |
| PlayGo | Libs | `xps5x-hle` | `todo` | | |
| pthread / threads | Kernel | `xps5x-runtime`/`hle` | `todo` | | M1-E |
| VideoOut / AGC / shaders→Vulkan | Graphics | `xps5x-gpu`/`hle` | `todo` | | M2+ |
| DualSense / pad | Input | `xps5x-input`/`hle` | `todo` | | |
| Filesystem: open/read/close/lseek | KernelExports/FS | `xps5x-kernel`/`hle` | `done` | `896495d` | VFS-backed, real host files under /app0; write persistence + fstat still todo |
| Filesystem / save (write persist, fstat, savedata) | FS | `xps5x-kernel`/`hle` | `todo` | | |
| GUI patterns | app | `xps5x-gui` | `skip` | | optional UX only |

**Delete SharpEmu when:** all non-`skip` rows `done`, and no open M# work still citing this tree.

---

## KytyPS5 (`reference/kytyps5`) — status: optional

| Area | Target | Status | Notes |
|------|--------|--------|-------|
| PS5 deltas over Kyty (SRT, pthread, VM, LibUlt, …) | merge into kyty-* / xps5x-* | `todo` | study; don’t blind-merge |
| Commercial boot paths | docs + HLE/GPU | `todo` | |

---

## shadPS4 (`reference/shadps4`) — status: optional

Pattern reference for Orbis HLE (memory, libkernel, linker). Port selectively; no need to 1:1 the whole tree.

| Area | Target | Status | Notes |
|------|--------|--------|-------|
| Memory model / libkernel patterns | `xps5x-hle`/`kernel` | `todo` | |
| Linker / NID ideas | `xps5x-firmware` | `todo` | |
| Vulkan present path ideas | `xps5x-gpu` | `todo` | |

---

## How to mark done + delete

1. Port module → tests green → set row `Status=done`, `Commit=<sha7>`.
2. When a reference has **zero** `todo`/`wip` rows left:
   - Set Index **Status** = `fully_ported`
   - Confirm `THIRD_PARTY_NOTICES.md` still credits upstream
   - `Remove-Item -Recurse -Force reference/<name>` (or `rm -rf`)
   - Set Index **Status** = `deleted`, note date in a one-line Log entry below

### Log

| Date | Action |
|------|--------|
| 2026-07-14 | Ledger created. Kyty + SharpEmu `active`. Seeded known done rows from SDD progress. |
