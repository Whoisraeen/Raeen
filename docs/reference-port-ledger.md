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
| KytyPS5 | `reference/kytyps5` | (check) | https://github.com/Nmzik/KytyPS5 | `not-cloned` | optional — clone only if a Kyty gap needs its PS5 deltas |
| shadPS4 | `reference/shadps4` | GPL-2.0 | https://github.com/shadps4-emu/shadPS4 | `not-cloned` | optional — clone only when consulting its Orbis HLE patterns |
| PS5SDK | `reference/ps5sdk` | GPL-2.0 | https://github.com/PS5Dev/PS5SDK | `not-cloned` | clone when building the M1 toolchain Hello World fixture |

> **Actual `reference/*` scope right now: `kyty` + `sharpemu` only** (the three
> rows above are not cloned — aspirational, and do not block the delete rule
> for the two present trees). The "delete when fully ported" condition applies
> only to trees that exist on disk.

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
| File | `kyty-core` | `skip` | | 1311-line buffered File class over sys_file_io — superseded by xps5x-kernel VFS on the hot path; verified the one ported consumer (json_reader) uses std, not Core::File, so nothing needs it. Port later only if a future Kyty subsystem does |
| SDLSubsystem | `kyty-core` | `skip` | | SDL window/input/audio init — superseded by xps5x-gui's eframe/egui (verified main.rs uses eframe) + xps5x-input/audio crates |
| Debug / Subsystems / Core.cpp | `kyty-core` | `todo` | | init glue over sys_dbg — port last (or skip: tracing + per-crate init supersede) |

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
| emulator/Graphics: texture micro-tiling | `xps5x-gpu` | `done` | `09f44c0` | fixed detile_micro from a bogus linear (py*8+px) interior to the documented GCN thin micro-tile Z-order (Morton interleave x0y0x1y1x2y2); added inverse tile_micro + round-trip/bijection/known-mapping tests. DEPTH/DISPLAY/ROTATED modes + macro bank/pipe swizzle + hardware-exact validation vs real dumps still todo |
| emulator/Graphics: PM4→shader→Vulkan pipeline | `kyty-graphics` → `xps5x-gpu` | `todo` | | crown jewel / M2+ — the big pipeline (needs real command-stream verification) |
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
| SELF / eboot loader: PT_SCE_PROCPARAM capture + GetProcParam wiring | Loader | `xps5x-firmware`/`runtime`/`kernel`/`hle` | `done` | `1787534`,`71221c5` | capture (SprxModule.procparam + proc_param_sdk_version); NOW wired end-to-end: LinkedModule.procparam_offset → runtime sets kernel.set_proc_param_addr(base+off) at load → sceKernelGetProcParam returns the real guest address (non-null sentinel only when the module has none). E2E tested |
| SELF / eboot loader: remaining (rebase, full param decode) | Loader | `xps5x-firmware` | `todo` | | non-zero load bias rebase, PT_SCE_MODULE_PARAM name extraction |
| PRX / sysmodule load chain | Kernel / Libs | `xps5x-firmware`/`hle` | `todo` | | |
| Fiber / AMPR | Kernel | `xps5x-hle`/`runtime` | `todo` | | |
| PlayGo | Libs | `xps5x-hle` | `done` | `39f7e58` | new libsce_playgo.rs: all chunks LOCUS_LOCAL_FAST, progress==total (complete), empty to-do list, handle out — "everything installed" so titles skip download gating (SharpEmu-cross-checked values). 13 NIDs; 3 tests |
| UserService (initial user / login list / name / event) | UserService | `xps5x-hle` | `done` | `3d44b94` | new libsce_user_service.rs: single local user id 1000 (SharpEmu PrimaryUserId), GetInitialUser/GetLoginUserIdList/GetUserName("Player")/GetEvent(NO_EVENT) — supplies the userId scePadOpen/save-data need. 6 NIDs; 3 tests |
| SystemService (status / param / safe-area) | SystemService | `xps5x-hle` | `done` | `46432e4` | new libsce_system_service.rs: GetStatus(eventNum=0, quiet), ParamGetInt (SharpEmu mapping 1/2/3/1000→1, 4→180), GetDisplaySafeAreaInfo(ratio 1.0), HideSplashScreen/ReportAbnormalTermination OK — a title's per-frame poll runs undisturbed. 5 NIDs; 3 tests |
| pthread / threads | Kernel | `xps5x-runtime`/`hle` | `todo` | | M1-E |
| AudioOut (pacing stub, no playback yet) | Audio | `xps5x-hle` | `done` | `bedb5e0` | libsce_audio_out.rs: Init/Open(handle+grain/freq)/Output(acks buffer, sleeps ~grain÷freq bounded, returns sample count)/Close/SetVolume — audio thread paces without hang or 100% spin (M3 "audio must not hang"). Real host playback (cpal) is the follow-up. 3 tests |
| VideoOut: flip-completion + resolution (no real present yet) | Graphics | `xps5x-hle` | `done` | `f908ddb` | SubmitFlip bumps a flip counter + records flipArg; GetFlipStatus reports count + zero-pending so the render loop advances (was stalling); GetResolutionStatus=1080p, GetVblankStatus=frame counter. Real swapchain present = M2/M3 follow-up. 2 tests |
| VideoOut real present / AGC / shaders→Vulkan | Graphics | `xps5x-gpu`/`hle` | `todo` | | M2+ — the actual GPU pipeline (crown jewel) |
| DualSense / pad (digital+analog state) | Input | `xps5x-input`/`hle` | `done` | `0ceb7db` | ControllerState→Orbis ScePadData encoder (documented button masks, stick/trigger byte mapping) in xps5x-input; scePadReadState writes a valid state + returns 1 (was garbage + 0 → homebrew read-loop hang). Live host-input routing (InputManager→HleContext) + haptics/adaptive-triggers still todo |
| DualSense: live input routing (kernel snapshot → scePadReadState) | Input | `xps5x-kernel`/`hle` | `done` | `cb1b56d` | kernel holds a settable 12-byte pad-state snapshot (OrbisKernel::set_pad_state/pad_state); scePadReadState reads it (neutral fallback when unset) — live host input now flows guest-ward. Remaining: Shell polling InputManager into set_pad_state each frame (UI wiring) + haptics/adaptive triggers |
| Filesystem: open/read/close/lseek | KernelExports/FS | `xps5x-kernel`/`hle` | `done` | `896495d` | VFS-backed, real host files under /app0; write persistence + fstat still todo |
| Filesystem: write persistence (savedata) | FS | `xps5x-kernel`/`hle` | `done` | `5285857` | VFS honors O_WRONLY/RDWR/CREAT/TRUNC/APPEND; write buffers + flush-on-close to host file; ".." traversal refused on writable open; hle write() routes non-console fds to VFS; hle open() honors O_CREAT. E2E: guest open+write+close persists to host, read-back works |
| Filesystem: fstat / directory ops | FS | `xps5x-kernel`/`hle` | `todo` | | needs SCE stat struct layout |
| SaveData mount | SaveData | `xps5x-hle` | `done` | `93a7c0a` | new libsce_save_data.rs (was empty stub): sceSaveDataMount{,2,3} writes the /savedata0 mount point into the 64-byte result; Umount/Initialize/Terminate OK. Completes the save path — mount → open/write under /savedata0 → VFS persists to host savedata dir (write-persistence 5285857). SharpEmu-cross-checked. 2 tests |
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
