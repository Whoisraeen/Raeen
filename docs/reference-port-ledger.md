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
| Sys (`sys_*.rs`, 9 mods) | `kyty-core` | `done` | `4ad2f49`,`b22c891` | were orphaned (drafted, never declared → never compiled); wired in `#[cfg(windows)]`, lint-fixed. Wiring exposed a STATUS_STACK_BUFFER_OVERRUN: SysCS drop transliterated Kyty's abort into a panic, leaving a live CRITICAL_SECTION on freed memory — fixed in b22c891 (Drop releases the OS resource). 92 tests live, both binaries exit clean. |
| CharUcd | `kyty-core` | `skip` | | use unicode crate; do not transliterate |
| Database | `kyty-core` | `skip` | | defer rusqlite unless needed |
| VirtualMemory (Core wrapper) | `kyty-core` | `done` | `e06b0d3` | `virtual_memory.rs` forwards 1:1 to sys_virtual (as Kyty's Core does on Windows); ExceptionHandler `skip` — xps5x-runtime VEH supersedes |
| MemoryAlloc / MSpace | `kyty-core` | `skip` | | manual C++ heap (`mem_alloc`/`mem_free` + stats) — superseded by Rust's global allocator (host) + `xps5x-runtime` `GuestArena` (guest); same rationale as skipped `SafeDelete`. Convention: manual-memory scaffolding → safe Rust equivalent. |
| Threads | `kyty-core`/`xps5x-runtime` | `todo` | | overlaps M1-E (real guest threads) — deferred, see SDD sketch |
| File | `kyty-core` | `todo` | | 1311-line buffered File class over sys_file_io; port only if a ported Kyty subsystem needs it (else xps5x-kernel VFS supersedes on the hot path) |
| Debug / Subsystems / SDL / Core.cpp | `kyty-core` | `todo` | | init glue, port last |

### Later Kyty trees

| Module | Target | Status | Commit | Notes |
|--------|--------|--------|--------|-------|
| lib/Math: VectorAndMatrix (Vec2/3/4, Mat2/3/4) | `kyty-math` | `done` | `f9ecddf` | `vector_and_matrix.rs` aliases Kyty vec/mat to `glam` (column-major, GNM/GLSL-order) + Kyty-named ctor helpers (splat/vec3_w/identity); 4 tests |
| lib/Math: Rand (mt19937) | `kyty-math` | `done` | `f9ecddf` | `rand.rs` — Kyty `Rand::*` API (uint/int/double/float + inclusive/exclusive ranges + seed) over the `rand` crate (StdRng, thread-local; not bit-identical to mt19937 — clean-room, sequence not load-bearing); 4 tests |
| lib/Math: Crypto (AES + Hash) | `kyty-math` | `skip` | | AES/SHA → RustCrypto (`aes`/`cbc`/`sha1` already workspace deps used by xps5x-firmware SELF decrypt); 3rdparty→workspace-crate convention, do not transliterate |
| lib/Scripts | `kyty-scripts` | `skip` | | Lua scripting — unused by XPS5X's execution path (guest games are native binaries, not Kyty Lua demos); per goal "skip unused Scripts/lua unless config needs it" |
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
