# Reference port ledger

**Goal:** port every useful module under `reference/*` into Raeen Rust crates.  
**Delete rule:** remove `reference/<name>/` only when that reference’s status is
`fully_ported` (all rows `done` or `skip`) and `THIRD_PARTY_NOTICES.md` still
attributes the upstream. Never delete mid-port.

**Status values (row):** `todo` · `wip` · `done` · `skip`  
**Status values (reference):** `active` · `fully_ported` · `deleted`

Update this file in the same session as each port batch. Link commit SHAs.

Claude `/goal` (≤200 chars):

```
/goal Port every useful module in reference/* into Raeen. Log status in docs/reference-port-ledger.md; delete a ref tree only when its ledger says fully ported.
```

---

## Index

| Reference | Path | License | Upstream | Status | Delete when |
|-----------|------|---------|----------|--------|-------------|
| Kyty | `reference/kyty` | MIT | https://github.com/InoriRus/Kyty | `active` | all rows done/skip |
| SharpEmu | `reference/sharpemu` | GPL-2.0 | https://github.com/par274/sharpemu | `active` | all rows done/skip |
| KytyPS5 | `reference/kytyps5` | GPL-2.0 + Kyty MIT lineage | https://github.com/Nmzik/KytyPS5 | `active` | all useful PS5 deltas done/skip |
| shadPS4 | `reference/shadps4` | GPL-2.0 | https://github.com/shadps4-emu/shadPS4 | `active` | all useful Orbis/Vulkan deltas done/skip |
| PS5SDK | `reference/ps5sdk` | GPL-2.0 | https://github.com/PS5Dev/PS5SDK | `not-cloned` | clone when building the M1 toolchain Hello World fixture |

> **Actual audited reference scope:** Kyty, SharpEmu, KytyPS5, and shadPS4.
> PS5SDK remains optional until the M1 compiler-produced fixture needs it.
> The "delete when fully ported" condition applies only to trees that exist on
> disk.

### SharpEmu refresh 2026-07-23 (559b7f0 → 6db095e, tag `v0.0.2-beta.5`, 21 commits)

Upstream landed **working AudioOut2 audio (GTA V)** + **AGC cross-queue label** work.
Extraction status:

- **AudioOut2 real playback** — **DONE (integrated 2026-07-23).** `libsce_audio_out2`
  now wires `ContextPush → AudioPcmConversion (new `crates/raeen-audio/src/pcm.rs`,
  s16/float→stereo) → `raeen_audio::output::submit` — the same cpal backend
  `libsce_audio_out` already uses — plus the canary (#532) / overflow (#564)
  hardening. Compiles (`cargo check -p raeen-hle -p raeen-audio` green). Refs:
  `Audio/AudioOut2Exports.cs`, `AudioPcmConversion.cs`.
- **Cross-queue `WAIT_REG_MEM` label producers** — **DONE (integrated + verified
  2026-07-23), HIGH value** — was Raeen's Minecraft render blocker (10+ labels whose
  producers never ran → dead-wait/force-resume → glitch, no menu progression).
  Ported the producer-packet execution into `kyty-graphics/src/run.rs`
  (`cp_op_write_data`/`cp_op_release_mem` now execute + write the label to guest
  memory; `IT_WRITE_DATA`/`IT_RELEASE_MEM` + `R_WRITE_DATA`/`R_RELEASE_MEM` arms
  rewired from "consumed without effect") plus the write-time latch in
  `raeen-gpu/src/agc_exec.rs` (`SuspendedBuffer.latched` + `latch_produced_waits`,
  survives the guest resetting a label the same frame). **Verified:** dead-waits
  75+→0, force-resumes 37→0, menu renders correctly, progresses to savedata init.
  SharpEmu `Agc/GpuWaitRegistry.cs` + `AgcExports.cs` (`LatchSatisfiedByValue`).
- **PR #587 Gen5 flat memory + 3D textures** — **DONE (decode integrated 2026-07-23).**
  `kyty-graphics/src/shader/{parse,recompile,types,resources,spirv,mod}.rs` now decode
  the PSSL SEG-field FLAT/global-address form (encoding 0x37, SEG bits [15:14]) into
  SPIR-V global-memory access via a `%global_mem` SSBO window, and carry the proper
  3D-texture path. Compiles combined with the cross-queue `run.rs`
  (`cargo check -p kyty-graphics` green). *Production host-window wiring of the flat
  shaders (shader_fetch.rs) is a follow-up — decode no longer dies, but the full
  render path for flat shaders is not yet exercised end-to-end.* Refs: SharpEmu #587.
### SharpEmu refresh 2026-07-24 (21f964a → 26c5029, 3 commits)

Two GPU commits + one audio. Both GPU commits were compared against Raeen
mechanically (table diff + algorithm simulation) rather than by inspection.

- **#592 `a158960` GPU compute detile** — **tiling math needs NOTHING.** Raeen's
  swizzle tables were verified **bit-for-bit identical** to SharpEmu's for every
  mode both implement: all 4 tables × 5 bpp rows × 16 address bits = 320 entries
  decoded to `(xmask, ymask)` pairs, **0 mismatches**; and both detile algorithms
  simulated on the 8 vectors from `GnmTilingDetileTests.cs` are **byte-identical**.
  SharpEmu's `x & XMask` is a lookup-table optimization, mathematically inert.
  **Corrects a stale claim** in `docs/rendering-blockers-and-port-plan-2026-07-22.md`
  ("Raeen's `tiling.rs` is CPU-only, 2 modes"): it has **4** modes (5/9/24/27) at
  5 element sizes. Bpp coverage is **equal**, not broader — SharpEmu's tables always
  had 5 rows; "4bpp" in that commit title refers only to its GPU kernel's scope.
  **Integrated:** row-parallel CPU detile (rayon, ≥512×512-element threshold,
  mirroring `GnmTiling.cs:533-539`) + a non-power-of-two element-size guard
  (`bpp_log2_is_supported`; callers derive bpp via `trailing_zeros`, so a 3-byte
  element silently read as 1 byte and a 32-byte element indexed **past** the last
  table row — a latent panic) + a rate-limited `(tile_mode, format)` diagnostic on
  the refusal path, so which modes titles actually use becomes a measurement.
  **Deliberately NOT ported:** the GPU compute pass. It is well-specified and
  Raeen's deferred-batch machinery fits it (~700-1000 lines), but it addresses
  *texture-upload* CPU cost, which is not in Tier 4's measured top-8 — the
  bottlenecks are per-flip readback (4.1) and per-submit fence waits (4.2). Do
  those first. **Queued, gated on the new diagnostic:** block-table modes 1/4/8
  (SharpEmu's own comment concedes it is a *model*, not a transcribed AddrLib
  PATINFO table — port only if a title is measured using them); mip-chain base
  placement (Raeen always reads `t.base40()`, so any `last_level > 0` texture
  would sample from the wrong offset — gate on measuring a title that binds one);
  a GPU-vs-CPU golden test (port as a `#[test]`, not a runtime env-var self-test).
- **#587 `5228335` Gen5 flat memory + 3D images** — the FLAT half was **already
  ported** in `c0f6303` (`parse.rs:3189-3309`, `recompile.rs:3696-3807`), and the
  3D depth-transport half is **already covered** (`draw_translate.rs:1263` ≡
  SharpEmu's `GetTextureVolumeDepth`, threaded to `TYPE_3D` + `extent.depth`).
  **Integrated:** `image_get_resinfo` no longer refuses non-2D descriptors. It
  hard-refused anything but plain 2D, which failed the **whole shader recompile**
  and dropped the draw — and it caught **2D-array as well as 3D**, i.e. every cube
  T# (types 11/13 lower to 2DArray). `OpImageQuerySizeLod`'s result width is fixed
  by the image's dim, so the query type now follows the descriptor (`%v3int` for
  3D/2DArray, `%v2int` for 2D); only x/y are stored either way. Added `%v3int` to
  the type preamble. **Deliberately NOT ported:** SharpEmu's rule that the MIMG
  `DIM` field *overrides* the descriptor. Raeen already decodes DIM
  (`parse.rs:3761-3769`) and uses it to gate whether a real third address VGPR
  exists, while taking the SPIR-V `Dim` from the T# nibble — descriptor-wins is
  defensible and already tested; SharpEmu's DIM-wins is unverified as more correct.
  **Queued:** `VkImageType` from the type nibble rather than `depth > 1`
  (`offscreen.rs:2972`, `compute.rs:1191`) — a type-10 T# with `depth()==0` gets
  SPIR-V `Dim3D` but a 2D image, the mismatch class blamed for an ASTRO.BOT device
  loss (unverified whether any title emits it); tiled 3D texture upload
  (`draw_translate.rs:1316` is an honest named refusal today, not a silent
  slice-0 under-read); tiled 3D UAV detile; FLAT D16 opcodes 0x19/0x1b/0x20-0x25.
- **#605 `26c5029` `sceAudioOutOutputs`** — not reviewed (out of this pass's scope).

**Adjacent gap found while comparing, bigger than anything in either commit:**
`texture_vk_format` (`draw_translate.rs:909-985`) has **no block-compressed format
arms at all** (no BC1/BC5/BC7). Most retail PS5 textures are BC. The 8/16-byte rows
of Raeen's swizzle tables are commented "also BC1/BC4 blocks" but are unreachable
for real BC textures. Not a SharpEmu port — a Raeen gap worth its own task.

- **Queued (moderate value):** kernel getdents-on-file-fd (#546), Posix `-1`/errno +
  open `EACCES` (#567), APR `ResolveFilepathsWithPrefixToIdsAndFileSizes` (#534);
  libc `cosf`/`time`/`ctype` tables (#542); Astro Bot AGC stack (#528, +520 lines to
  `AgcExports`: swapchain fallback, title clear). Port when the relevant subsystem needs it.

### Minecraft graphics/audio/input closure slice 2026-07-24

- **GFX10 MIMG DIM/NSA and writable arrays — DONE (measured).** The active
  `kyty-graphics` parser now decodes GFX10 `DIM` and NSA extension DWORDs,
  accounts for their real instruction length, and supplies explicit address
  VGPRs to sample/load/store lowering. Writable cube/2D-array descriptors use
  arrayed SPIR-V coordinates and honor `BASE_ARRAY..=LAST_ARRAY`; graphics and
  compute storage-image uploads/writeback preserve every layer and its guest
  tiling. GFX10 `VCMPX` now intersects EXEC without incorrectly overwriting
  VCC. The instruction semantics were cross-checked against SharpEmu's RDNA2
  tables and KytyPS5's shader path; the Rust lowering and regression fixtures
  are original.
- **Guest cubemap `(s,t,face)` lowering — DONE (measured Minecraft panorama).**
  Type-11 T# descriptors now share the 2D-array SPIR-V/Vulkan path with type 13.
  The guest's `V_CUBE{SC,TC,MA,ID}` sequence has already converted a direction
  into face coordinates; the former Vulkan `Cube` view interpreted those
  values as a second direction and produced radial face smearing. Cross-checked
  against SharpEmu/KytyPS5 and pinned by shader plus six-layer upload tests.
- **Resource-class-local storage indices — DONE (measured).** A Minecraft
  compute shader reused `s24` for a storage descriptor and a later sampled
  descriptor. Dynamic use of the rewritten descriptor DWORD selected sampled
  slot 1 from a one-element storage array, silently discarding every
  `ImageStore`. Storage image access now resolves its exact analyzed descriptor
  to a class-local constant index. Six panorama cube faces write non-zero guest
  data in a real run.
- **Cube upload safety — DONE (measured).** CPU render-target snapshots may
  replace only plain, one-layer 2D uploads. A framebuffer image can no longer
  replace a cube/array/volume at an aliased base while retaining the larger
  layer count. The Vulkan create sites independently size and zero-pad sampled
  staging bytes from the declared extent/format/layers. This removes the
  measured Minecraft validation failure
  `VUID-vkCmdCopyBufferToImage-pRegions-00171` (24 MiB six-face copy from a
  4 MiB staging buffer) and prevents the associated device reset.
- **AudioOut2 production ABI — DONE for Minecraft's measured PCM route.**
  SharpEmu's Gen5 context/port layout and pacing were adapted to Raeen's HLE
  memory model; ACM/media descriptor outputs are initialized, and the isolated
  title runner starts the same cpal host output as the Shell. A release run
  produced non-silent 48 kHz PCM on the host.
- **DualSense route — title consumption DONE; interaction acceptance OPEN.**
  SharpEmu and KytyPS5 both deliver an initial UserService login event before
  returning `NO_EVENT`; shadPS4 likewise queues login events. Raeen previously
  returned permanent `NO_EVENT`, so Minecraft never entered its Pad path.
  UserService now uses the retail-style primary id `0x10000000` and a
  process-scoped one-shot login event. The isolated
  runner polls Raeen's native XInput/raw-HID backend and publishes the full
  120-byte Orbis pad state through `libScePad`; the layout is cross-checked
  against shadPS4 and KytyPS5. A release run measured a real raw-HID DualSense
  report, login-event consumption, Pad handle 1, and Minecraft's first guest
  `scePadReadState` consuming the live host sticks. Do not claim gameplay input
  until a button-driven menu transition is also captured.

### Remaining-work classification (as of the latest batch)

Every **self-contained, verifiable** module in both references is `done`/`skip`
— including *every* SharpEmu-implemented libSce. What keeps both trees `active`
(not yet `fully_ported`) is exclusively work that needs a **subsystem stood up**
or a **real fixture**, none of which can be honestly completed as a stub:

1. **M2 GPU draw infrastructure (synthetic proof; gate OPEN)** — Fixture Gen5 AGC PM4
   `DRAW_INDEX_AUTO` → `agc::decode_submission` → `AgcGpuSession` → Vulkan
   offscreen draw using `kyty-graphics` SPIR-V (`shader_bridge`: VS via
   `spirv_asm`, FS via full GCN→recompile), proven by pixel readback + PPM in
   `tests/m2_agc_triangle.rs` (`RAEEN_REQUIRE_VULKAN=1`). HLE
   `sceAgcDriverSubmitDcb` hooks the same session. This synthetic fixture does
   not satisfy Raeen's acceptance-gate by itself. **Still open for title
   frames / M2+:** full Kyty `GraphicsRun` register-state PM4 processor,
   guest shader bind from SH regs, and VideoOut → swapchain present.
2. **M1-E execution contexts** — Kyty `Threads`, SharpEmu `pthread`, `Fiber`,
   `AMPR`. All need real guest execution on a fresh context; an init-only stub is
   the "no-op instead of real execution" anti-pattern (verified for Fiber).
3. **Fixture-blocked loader** — non-zero load-bias rebase, real user `.prx` chain.
   Need a real toolchain-built `.sprx` (the `not-cloned` `PS5SDK` fixture row) to
   verify against rather than guess.

These are the same items as the M2 / M1-E / loader milestones — i.e. finishing the
port == finishing the emulator's remaining big milestones. Not stub-shaped work.

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
| VirtualMemory (Core wrapper) | `kyty-core` | `done` | `e06b0d3` | `virtual_memory.rs` forwards 1:1 to sys_virtual (as Kyty's Core does on Windows); ExceptionHandler `skip` — raeen-runtime VEH supersedes |
| MemoryAlloc / MSpace | `kyty-core` | `skip` | | manual C++ heap (`mem_alloc`/`mem_free` + stats) — superseded by Rust's global allocator (host) + `raeen-runtime` `GuestArena` (guest); same rationale as skipped `SafeDelete`. Convention: manual-memory scaffolding → safe Rust equivalent. |
| Threads | `kyty-core`/`raeen-runtime` | `todo` | | overlaps M1-E (real guest threads) — deferred, see SDD sketch |
| File | `kyty-core` | `skip` | | 1311-line buffered File class over sys_file_io — superseded by raeen-kernel VFS on the hot path; verified the one ported consumer (json_reader) uses std, not Core::File, so nothing needs it. Port later only if a future Kyty subsystem does |
| SDLSubsystem | `kyty-core` | `skip` | | SDL window/input/audio init — superseded by raeen-gui's eframe/egui (verified main.rs uses eframe) + raeen-input/audio crates |
| Debug / Subsystems / Core.cpp | `kyty-core` | `skip` | | verified: Subsystems=a dependency-ordered init/shutdown manager (superseded by Raeen's per-crate `new()` init, no central manager); Debug/DebugMap=C++ symbol-map (.map/.csv/MSVC) loading for backtrace symbolication (superseded by Rust's native backtraces + `tracing`); Core.cpp=Core::Init glue. All init-glue/scaffolding Raeen's architecture replaces. sys_dbg (the substantive Sys-layer piece) already ported. |

### Later Kyty trees

| Module | Target | Status | Commit | Notes |
|--------|--------|--------|--------|-------|
| lib/Math: VectorAndMatrix (Vec2/3/4, Mat2/3/4) | `kyty-math` | `done` | `f9ecddf` | `vector_and_matrix.rs` aliases Kyty vec/mat to `glam` (column-major, GNM/GLSL-order) + Kyty-named ctor helpers (splat/vec3_w/identity); 4 tests |
| lib/Math: Rand (mt19937) | `kyty-math` | `done` | `f9ecddf` | `rand.rs` — Kyty `Rand::*` API (uint/int/double/float + inclusive/exclusive ranges + seed) over the `rand` crate (StdRng, thread-local; not bit-identical to mt19937 — clean-room, sequence not load-bearing); 4 tests |
| lib/Math: Crypto (AES + Hash) | `kyty-math` | `skip` | | AES/SHA → RustCrypto (`aes`/`cbc`/`sha1` already workspace deps used by raeen-firmware SELF decrypt); 3rdparty→workspace-crate convention, do not transliterate |
| lib/Scripts | `kyty-scripts` | `skip` | | Lua scripting — unused by Raeen's execution path (guest games are native binaries, not Kyty Lua demos); per goal "skip unused Scripts/lua unless config needs it" |
| emulator/Loader | → `raeen-firmware`/`raeen-loader` | `wip` | (M0/M1) | realized on the hot path: SELF decrypt-or-passthrough → sprx parse → PT_SCE_DYNLIBDATA/PT_DYNAMIC decode → NID link (LM0/LM1), + PT_TLS + PT_SCE_PROCPARAM capture. Remaining: non-zero load-bias rebase, real user .prx chain, full param decode — see loader `todo` rows above |
| emulator/Kernel | → `raeen-kernel`/`raeen-hle` | `wip` | (M1) | realized: OrbisKernel (VMM, VFS, thread mgr, module table, console, proc-param, pad state) + libkernel HLE (mem/time/module/proc-time/fd/file I/O). Remaining: real threading (M1-E), Fiber/AMPR, broader syscall surface |
| emulator/Libs (libSce* HLE surface) | `raeen-hle` | `wip` | (many) | realized as raeen-hle libSce* modules: libc, libkernel (mem/time/module/proc-time), Sysmodule, PlayGo, User/System Service, Pad, AudioOut, VideoOut, SaveData, CommonDialog/MsgDialog, AppContent, Np, Net/NetCtl, DiscMap, **Rtc**. The high-traffic boot/HUD libs are done, plus Rtc + Mouse/Ime/GameUpdate; the media-subsystem handshakes (Ajm/Ngs2/AvPlayer/Ult) are now ported too (as handshake stubs, matching SharpEmu — no real output yet). **Every SharpEmu libSce that isn't GPU-command or scheduler infrastructure is now done.** **Agc** (`libsce_agc.rs`, 51 of ~62 clean register-args functions ported `67d3399`…`42327c2`) and **Ampr** (`libsce_ampr.rs`, command-buffer lifecycle + MeasureCommandSize + ClearBuffer + `AprCommandBufferReadFile` missing-file zero-fill, 13 NIDs, `0f972fc`,`fbfe705`) are now substantially ported. The `Rsp`-stack-arg ABI limit is resolved (`7465245` — dispatch now captures SysV args 7+); the four Agc sync/DMA emitters (`ef5bea5`) and Ampr ReadFile (`fbfe705`) are done. The residue is register-defaults tables (RDNA2 data) and the file-registry-backed ReadFile / command-*content* writers, all of which land with the Vulkan/compute + I/O backend. **Fiber**: its config/profiling calls (`OptParamInitialize`, `Start/StopContextSizeCheck`) are ported (`libsce_fiber.rs`, `39a6d69`); only `Run`/`Switch`/`ReturnToThread`/`Initialize` (control transfer into a fiber's guest entry) remain — the M1-E scheduler gate, deliberately not stubbed. Real audio/video *output* for AudioOut/Ngs2/Ajm/AvPlayer is a follow-up (host backend), not a port gap. GraphicsDriver = the M2 pipeline row. |
| emulator/Graphics: texture micro-tiling | `raeen-gpu` | `done` | `09f44c0` | fixed detile_micro from a bogus linear (py*8+px) interior to the documented GCN thin micro-tile Z-order (Morton interleave x0y0x1y1x2y2); added inverse tile_micro + round-trip/bijection/known-mapping tests. DEPTH/DISPLAY/ROTATED modes + macro bank/pipe swizzle + hardware-exact validation vs real dumps still todo |
| emulator/Graphics: PM4 command decoder | `raeen-gpu` | `wip` | `9cee071` | first verified slice of the pipeline: added tests for process_command_buffer (Type 0 reg write, Type 3 SET_CONTEXT_REG + DRAW/DISPATCH, Type 2 NOP) — decode + register state + stats asserted against hand-built PM4 buffers. Found an API smell (registers write_context takes an offset but read_context takes an absolute addr — bug-prone, flagged). Remaining: full opcode coverage, shader (GCN→SPIR-V), Vulkan submit, real present |
| emulator/Graphics: SPIR-V emitter (header/structure) | `raeen-gpu` | `wip` | `ddedfab` | verified the SPIR-V emitter produces structurally-valid modules for every shader stage: correct magic (0x07230203) + version + 5-word header + nonzero id-bound, pixel shader emits OpExecutionMode(OriginUpperLeft), geometry shader declares the Geometry capability. Remaining: real IR→SPIR-V body emission (currently a minimal main{ret}), GCN decode→IR, Vulkan submit + present |
| emulator/Graphics: SPIR-V constant pool | `raeen-gpu` | `wip` | `2f51472` | `emit_spirv` declares an i32 scalar type and materializes each distinct IR constant as an `OpConstant` (ints on i32, floats on f32), deduped by (type, bits) — the first real body-emission step past `main{ret}`, producing the ids the arithmetic body will reference. Verified by parsing the emitted module back: constant present, emitted once for a repeated value, absent when the program has none |
| emulator/Graphics: rspirv structural validation gate | `raeen-gpu` | `done` | `df1f661` | added `rspirv` 0.12 as a **test-only** dev-dependency (MIT license option — GPL-2.0-only compatible, not linked into the emulator; attributed in `THIRD_PARTY_NOTICES.md` `42899be`). Every emitted module is now parsed via `rspirv::dr::load_words` — real structural validation (magic/version, per-instruction word counts, operand layout) across all six stages + a constant-pool module. User-approved dependency choice (rspirv over Apache-2.0 spirv-val). |
| emulator/Graphics: SPIR-V arithmetic body | `raeen-gpu` | `wip` | `8ce7dc8` | `emit_spirv` lowers the IR body: walks nodes in SSA order, resolves sources to constant-pool ids / prior SSA result ids / type-correct shared `OpUndef` (unwired live-ins), and emits `OpFAdd/FSub/FMul/FDiv` + `OpIAdd/ISub/IMul` with fresh result ids threaded to later uses. Verified via rspirv's **parsed** module: `r0=2.0+3.0; r1=r0*r0` → an `OpFAdd` then an `OpFMul` referencing the add's result; whole module parses. Remaining: real I/O interface vars (`OpVariable`+`OpLoad`/`OpStore` replacing the undefs) and full `spirv-val` semantic validation, then Vulkan submit + present |
| emulator/Graphics: SPIR-V logical-layout + input interface vars | `raeen-gpu` | `wip` | `30de312` | **(a) Layout correctness:** the builder emits into per-section buffers (caps → memory-model → entry-points → exec-modes → annotations → types/vars → functions) concatenated in SPIR-V's mandated order; previously types preceded `OpEntryPoint` (rspirv tolerated it, `spirv-val` would not). Test asserts `OpEntryPoint` precedes the first type decl. **(b) Inputs:** each distinct `IrValue::Input(loc)` becomes an Input-storage `OpVariable` (ptr-to-f32) with `OpDecorate Location`, listed in the entry-point interface and `OpLoad`-ed once (cached) in the body — replacing the input `OpUndef`s. Verified via rspirv's parsed module (2 inputs → 2 Input vars, both in interface, body OpLoads + OpFAdd). |
| emulator/Graphics: SPIR-V output interface vars | `raeen-gpu` | `wip` | `d153bd0` | each export node (`ExportColor/Position/Param`) declares an Output-storage `OpVariable` (ptr-to-f32) at a successive Location, listed in the entry interface, and the body `OpStore`s the export's resolved value into it. **Shader I/O path now complete: inputs `OpLoad` → arithmetic → exports `OpStore`, in correct logical layout, rspirv-validated** (`r0=in0+in1; export r0` → 1 Output var in interface + an OpStore). Remaining: int/vec4 I/O types, position/param builtin decorations, full `spirv-val` semantic pass, then Vulkan submit + present |
| emulator/Graphics: GCN instruction decoder | `raeen-gpu` | `wip` | `3acad0f` | verified the RDNA2 decoder: encoding classification (VOP2/VOP3/SOPP) + instruction widths (4/8B), 8-byte raw assembly (word1<<32\|word0), stream walk with byte-offset advance, and stop-at-S_ENDPGM. <4-byte binary errors. Remaining: full operand (src/dst) decode + the GCN→IR lowering that feeds the SPIR-V body |
| emulator/Graphics: GCN→IR lowering (encoding→IrOp) | `raeen-gpu` | `wip` | `0e11786` | verified `lift_to_ir`: the encoding→IrOp table (VOP2 Add/Mul/IAdd, VOP1 Mov/Sqrt, SOP2/SOP1), sequential SSA result numbering, EXP/S_ENDPGM as sinks with no SSA result, and resource counting (SMEM→ubo, MIMG→texture, VINTRP→input, EXP→output). Unknown opcode within a known encoding → Nop (no panic). Remaining: operand (src/dst) wiring into IR `sources`, which then unblocks real SPIR-V body emission |
| emulator/Graphics: VOP1/VOP2 operand decode | `raeen-gpu` | `wip` | `d0d9c7d` | `decode_operands` fills `Instruction.src/dst` for VOP1/VOP2: the 9-bit SRC0 field (`decode_src9` — SGPR 0-101, VGPR 256-511→0-255, inline int +1..+64/-1..-16, inline float 0.5/±1/±2/±4 as IEEE bits, VCC/EXEC/M0, literal-follows marker), VSRC1, VDST. 4 tests vs known encodings. Unmodeled encodings keep empty operands (honest partial coverage). Remaining: SMEM/MIMG/EXP operand layouts + SSA-correct `sources` wiring (vgpr→last-def map) |
| emulator/Graphics: GCN→IR SSA source wiring | `raeen-gpu` | `wip` | `9c96214` | `lift_to_ir` threads VOP operands into IR `sources` by local value numbering: a VGPR read resolves to the SSA reg a prior op wrote (`vgpr_def`), an undefined VGPR read is a live-in `Input`, inline int/literal fold to IR constants; each VOP result records its VDST → real def→use chains (verified `v2=v0+v1; v3=v2*v2`). Remaining: a parallel scalar (SGPR) SSA map, SMEM/MIMG/EXP operand decode, then IR→SPIR-V arithmetic body emission (the SPIR-V emitter's `main{ret}` becomes real once sources reach it) |
| emulator/Graphics: scalar (SGPR) SSA path | `raeen-gpu` | `wip` | `90128c2` | `decode_operands` now decodes SOP2 (SSRC0/SSRC1/SDST) + SOP1 (SSRC0/SDST) 8-bit scalar fields; `lift_to_ir` gains a `sgpr_def` value-numbering map symmetric to the vector one, and `resolve_source` consults both (replacing the SGPR→Input approximation). Verified scalar def→use chain (`s2=s0+s1; s3=mov s2`). Remaining: SMEM/MIMG/EXP operand layouts, then IR→SPIR-V arithmetic body — which is gated on a `spirv-val`/spirv-tools validator in the test harness (structure is testable in-tree; full body *validity* is not, so it won't be emitted-and-claimed without one) |
| emulator/Graphics: EXP export-target decode | `raeen-gpu` | `wip` | `0eb504a` | `classify_instruction` carries the EXP `TGT[9:4]` as the opcode; `lift_to_ir` maps it to the export kind (MRT0-7/MRTZ→`ExportColor`, POS0-3→`ExportPosition`, PARAM0-31→`ExportParam`) instead of a hardcoded color export. 3 targets verified. Remaining: the four `VSRC` export operands live in word1 → needs the 64-bit `raw` threaded into operand decode |
| emulator/Graphics: EXP VSRC operand decode | `raeen-gpu` | `wip` | `902ed97` | `decode_operands` now takes the full 64-bit `raw`; the EXP arm reads `EN[3:0]` (word0) and pulls the enabled `VSRC0..3` VGPRs from word1, and `lift_to_ir` threads them through the SSA maps so an exported VGPR references the value that produced it (verified `v2=v0+v1; export v2` → source is the add's SSA reg; partial EN masks select the right subset). Vector/scalar ALU + EXP now all carry real SSA operands. Remaining front-end: SMEM/MIMG operand layouts, then IR→SPIR-V arithmetic body (gated on a `spirv-val` validator in the harness) |
| emulator/Graphics: SMEM operand decode | `raeen-gpu` | `wip` | `d563a77` | `decode_operands` SMEM arm decodes `SBASE[5:0]` (descriptor SGPR-pair base, ×2) as a source and `SDATA[12:6]` as the destination SGPR; `lift_to_ir` resolves the descriptor through the SSA maps and records the `BufferLoad` result into `sgpr_def` so a later scalar read of that SGPR chains to the load. Verified field extraction. **Front-end now decodes + SSA-threads VOP1/2, SOP1/2, EXP, SMEM.** Remaining: SMEM offset (SOFFSET/imm in word1), MIMG/MUBUF/MTBUF operand layouts, then IR→SPIR-V arithmetic body (gated on a `spirv-val`/spirv-tools validator in the test harness) |
| emulator/Graphics: PM4 command **encoder** (`Pm4Writer`) | `raeen-gpu` | `done` | `db50b13` | in-house PM4 build side — the exact inverse of `Pm4Header::from_raw`/`process_command_buffer`. `set_context_reg`/`set_sh_reg`/`draw_index_auto`/`dispatch_direct`/`write_type0`/`nop`/`type3`. Round-trip verified: encoder output decodes to the right register values + draw/dispatch/packet counts. This is the GPU-command-emission foundation the `libSceAgc` port (66 DCB/ACB PM4 emitters) builds on — kept as clean round-trip-verified code rather than a low-confidence transliteration of Agc's command-buffer object model. 3 tests. |
| emulator/Graphics: libSceAgc DCB emitters (cursor + PM4) | `raeen-hle` | `wip` | `67d3399`,`1745228`,`217d4f7`,`3a8d787`,`ee492b2`,`3653788`,`3620a10`,`c496e06`,`c54561e`,`dbbd5e4`,`7545443`,`8a4f09e`,`0c28932`,`30eef6b`,`1b5d896`,`10ac4fa`,`7f43378`,`474c70e`,`4b894c8`,`ef489da`,`951c2cf`,`6af4a14`,`f705b58`,`42327c2`,`7465245`,`ef5bea5` | `libsce_agc.rs` — port of SharpEmu Agc's Draw-Command-Buffer cursor model (`alloc_command_dwords`: CB struct cursor-up@0x10/down@0x18/reserved@0x30) + **15 functions**: `Init` + 19 command emitters + `CreatePrimState`+`CreateShader`+`CreateInterpolantMapping`+`Set{Cx,Sh,Uc}RegIndirectPatch*`(6)+`GetDataPacketPayloadAddress`+`DcbDispatchIndirect`+`Driver*`+`SuspendPoint`+`DcbDrawIndexOffset`+`Unknown`+`PushMarker`+`DriverInitResourceRegistration`+`AcbEventWrite`+`SetBaseIndirectArgs`+`DriverSubmitDcb/Acb`+patch-address-setters(3)+`SubmitMultiDcbs`+`QueryResourceMemory`+driver-owner/eq-events(4)+`DriverUnknown` across DCB/Cb/Acb (draws, index setup, instances, markers, dispatch, flip, resets, nop, wait-flip, SH-register-range copy, event-write), emitting PM4 in Agc's dialect (`0xC000_0000\|(len-2)<<16\|op<<8\|(reg&0x3F)<<2`). Verified vs SharpEmu's exact bytes + cursor advance. 27 tests. **Verifiable boundary:** every Agc function that's a clean, fixed-layout packet emitter is verifiable vs exact bytes. **Correction:** struct-writers whose offsets the reference specifies (e.g. `CreatePrimState`, `7545443`) ARE portable + testable-vs-reference, and are done. **Stack-arg ABI limit RESOLVED (`7465245`):** the runtime HLE dispatch now captures SysV on-stack args 7+ at `[Rsp+8+i*8]` into `args[6..]`, so the variable-length sync/DMA emitters that overflow past the 6 register args are now portable + byte-verifiable. Ported (`ef5bea5`): `DcbWriteData` (WRITE_DATA, inline-dword copy; increment/writeConfirm = args 7-8), `DcbWaitRegMem` (WAIT_REG_MEM / 32-bit / 64-bit poll forms; reference/mask/pollCycles = args 7-9), `DcbDmaData` (DMA_DATA; control4/sourceAddress/byteCount/control7-9 = args 7-12), `DcbAcquireMem` (ACQUIRE_MEM; pollCycles = arg7, `u64::MAX` no-size sentinel). Also fixed a mismap: `sceAgcAcbWriteData` (NID `eZ4+17OQz4Q`) aliases `DcbWriteData` in the reference and was wrongly bound to the dispatch-indirect emitter (which is `sceAgcAcbDispatchIndirect`, NID `j3EtxFkSIhQ`). What genuinely stays blocked is only the large hardware **register-defaults tables** (`GetRegisterDefaults2*`) — unverifiable RDNA2 data. That + `Ampr` command-*content* writers land properly with the Vulkan submit path (real consumer to verify against) — not as guessed transcriptions. |
| emulator/Graphics: GCN→IR body + Vulkan submit→present | `kyty-graphics` → `raeen-gpu` | `wip` | M2 triangle | **Synthetic M2 infrastructure proof; gate open:** `shader_bridge` wires `kyty-graphics` (VS `spirv_asm` + FS GCN recompile) into Vulkan; `agc_exec::AgcGpuSession` runs on DRAW packets; HLE `SubmitDcb` calls it; `tests/m2_agc_triangle.rs` asserts pixels + writes PPM. **Still todo for titles and M2 acceptance:** Kyty `GraphicsRender` (real indexed/indirect draws, guest shader bind), non-synthetic input, and VideoOut swapchain present. GraphicsRun's CP itself is expanded — see the `GraphicsRun.cpp` CommandProcessor row. |
| emulator/Graphics: `GraphicsRun.cpp` CommandProcessor — Gen5 op coverage + retail-DCB resilience | `kyty-graphics` → `raeen-gpu` | `wip` | (this session, pending commit) | `run.rs` expanded for real title DCBs. **(1) Resilience policy:** unknown IT opcodes / AGC custom ops / registers no longer kill the DCB — rate-limited warn (once per distinct op per CP instance, `distinct_skips()` diagnostic), skip by encoded dword length, continue; hard `CpError` reserved for structural faults (`Truncated`, `NotType3`) and refused draws (`Draw`). **(2) Gen5 micro-ops ported from Kyty:** `R_DRAW_INDEX` (cp_op_draw_index L2757, both AGC 0xC008100C and raw `IT_DRAW_INDEX_2` 0xc0042700 forms), `R_{CX,SH,UC}_REGS_INDIRECT` (L3018/3050/3082 — guest `(offset,value)` pairs feed the same per-register setters as direct writes), `R_DRAW_RESET` → `Reset` (L519; warn rate-limit deliberately survives), `IT_INDEX_TYPE`/`SetIndexType`, plus `IT_INDEX_BASE`/`IT_INDEX_BUFFER_SIZE`/`IT_SET_BASE(1)` state tracking and rate-limited no-op skips for acquire/release-mem, event-write(+EOP/EOS), wait-reg-mem, write-data, flip, wait-flip-done. **(3) Guest memory behind a trait:** new `GuestMemory` + `run_with_memory`; Kyty's raw pointer derefs replaced; `raeen-gpu::guest_mem::IdentityGuestMemory` implements it over the identity map with VirtualQuery-validated reads (SAFETY-documented), wired into `AgcGpuSession::execute_dcb_cp`. **(4) Indexed/indirect draws degrade honestly:** `DrawSink::draw_index` default = vertex-count-only auto draw (indices NOT fetched — right count, wrong order for non-sequential indices); indirect draws read only the first args record at `SET_BASE+offset` for a count (multi stride/count not walked); all degradations logged rate-limited. Real index fetch + per-draw state = GraphicsRender (still todo). 194 kyty-graphics + 86 raeen-gpu tests green. |
| emulator/Graphics: **Vulkan 1.3 backend — device + offscreen draw + pixel readback** | `raeen-gpu` | `partial` | `7b1e68c` + synthetic proof | Device + offscreen + readback (`7b1e68c`); the synthetic path uses `render_triangle_with_spirv` + `render_m2_triangle` (kyty SPIR-V). Hand modules in `vulkan/shaders.rs` remain for backend smoke only. Real swapchain presentation and non-synthetic acceptance evidence remain open. |
| emulator/Graphics: SPIR-V assembly-text assembler (`spirv_asm`) | `kyty-graphics` | `done` | `3eadb8c`, wired `5eb7485` | pure-Rust replacement for the SPIRV-Tools *Assemble* step of Kyty's `SpirvRun` (Shader.cpp L845, `SPV_ENV_VULKAN_1_2` L850 → SPIR-V 1.5 header): `assemble(text) -> Vec<u32>` covering the exact vocabulary `ShaderSpirv.cpp` emits — **123 opcodes** (all 114 Kyty-emitted + 9 structural: OpName/OpSource/OpSpecConstant*/OpUnreachable/OpDemoteToHelperInvocation), **102 enum tokens** across 16 operand kinds, **32 GLSL.std.450 names** + NonSemantic.DebugPrintf-by-number (L6167). Handles Kyty's numeric/symbolic id mix (`%4` vs `%void`, EMBEDDED_SHADER templates L1279+), forward refs, typed 32-bit constants (dec/`0x%08x`/`%f`, L6527-6535), `Lod` image-operand masks, default-only OpSwitch (L252), atomics with scope/semantics as ids (L2230). 30 tests incl. hand-computed golden words + `naga` (spv-in dev-dep) parse gates on a Kyty-shaped fragment shader. **Ledger correction:** the old "NOT yet wired into shader/ recompiler output path" note was stale — `5eb7485` landed the wiring after that row was written. `spirv_run` (recompile.rs) *is* `Ok(spirv_asm::assemble(source)?)`, and `shader_recompile_{vs,ps,cs}` all route through it. Kyty's Validate/Optimize passes are deliberately not ported (assemble-only); naga covers validation in tests |
| emulator/Graphics: GCN shader parser + data model (`ShaderParse.cpp`/`Shader.h`) | `kyty-graphics` | `done` | `3eadb8c` | faithful port: full Shader.h data model (198-variant `ShaderInstructionType`, 52 packed-u64 `Format` constants via `format_define`, modifier-ignoring `ShaderOperand` equality, labels/CFG `read_block`/`read_intructions` sic) + `operand_parse` (canonical operand-code map incl. all 8 inline floats, literal-in-next-dword) + all 17 encoding-family parsers (SOPC 12, SOPK 2, SOPP 10, SOP1 6, SOP2 26, VOPC 39+SDWA, VOP1 23, VOP2 29, VOP3 89 w/ legacy-vs-next-gen abs/neg/omod layouts, EXP, SMEM 5 w/ 21-bit signed offset, SMRD 7, MUBUF 8, DS 2, MIMG 6 dmask→width, MTBUF 2, VINTRP 3) + end detection (0xBF810000 w/ live-label exception, 0xBE802000 fetch). Deviation: typed `ShaderParseError` + tracing instead of Kyty hard-EXITs; 512-case fuzz proves no panics. 53 tests |
| emulator/Graphics: shader analysis layer (`Shader.cpp` L37-3004) | `kyty-graphics` | `done` | `bf67085` | binary-info/usage-slot discovery (0xBEEB03FF sentinel @ code+(code[1]+1)*2, backwards usage_masks walk), V#/T#/S#/GDS/EUD sharp decode from user SGPRs w/ hand-computed bitfield tests, external fetch-shader parse (SLoadDwordx4↔BufferLoadFormat* matching) + PS5 embedded-attrib path (`ShaderParseUsage2` sharp tables), `ShaderDetectBuffers` V#-merge, `ShaderCalcBindingIndices`, `ShaderGetId*` cache keys (layout-not-contents, regression-tested), `shader_parse_vs/ps/cs` wrappers (next-gen hash from gs/ps_regs.chksum), minimal `hw_regs.rs` HardwareContext slice. Deviations: bounds-checked `ShaderMemory` trait for guest reads, `ShaderMap` struct instead of `g_shader_map` global, typed `ShaderAnalysisError`. 45 tests (128 crate total). Remaining: debug dumps (L520-900, L1842-2285), disabled-shaders/printf-inject dev tooling (L2967-3006) |
| emulator/Graphics: SPIR-V generator (`ShaderSpirv.cpp` GenerateSource) + **full chain closed** | `kyty-graphics` | `wip` | `5eb7485`, chain proven `3f74e52` | `spirv.rs` (`spirv_generate_source`) emits Kyty's SPIR-V **assembly text**; `recompile.rs` holds the `G_RECOMP_FUNC` dispatch table (204 rows, mirroring Kyty row-for-row) + `spirv_run` → `spirv_asm::assemble` → words. **The GCN→validated-SPIR-V chain is now closed and proven end-to-end** by `tests/gcn_to_spirv.rs`: real GCN bytes wrapped in a genuine `0xBEEB03FF` binary-info blob → `shader_parse_{vs,ps}` → **analysis** (`shader_get_input_info_{vs,ps}`, deriving input info from hardware registers rather than hand-built) → `shader_recompile_{vs,ps}` → `spirv_asm::assemble` → **`naga` parse + `Validator::validate`** + single-entry-point/stage assertion. Composed exactly as Kyty's `GraphicsRender.cpp` L2636-2639 — Kyty has **no** single combined entry point, so no such API was invented; the test is the composition site until GraphicsRender lands. **Named boundary (what this does NOT cover):** (a) the dispatch table implements **77 of 204** rows; the other **127 are `NotImplemented`**, each naming its Kyty function + line. (b) `48f653f` staged **51** further recompiler fns (image sample/load/store, buffer/tbuffer stores, S_* 64-bit, V/S compares, cvt_pkrtz, mbcnt, ds append/consume) that are **written but not wired** into `G_RECOMP_FUNC` — marked `#[allow(dead_code)] // C2: staged recompiler, not yet wired`; wiring each needs its `(type, format)` row flipped **plus a per-opcode test**, deliberately not done blind. (c) Kyty's Validate/Optimize passes not ported. (d) Only VS/PS proven; CS untested. **Also fixed here:** `48f653f` left `main` **not compiling** (13 uses of `operand_load_int`, never imported) — restored |
| emulator top (Audio/Controller/…) | `kyty-emulator` | `todo` | | |

**Delete Kyty when:** every row above is `done` or `skip`, crates wired into hot path or explicitly N/A.

---

## SharpEmu (`reference/sharpemu`) — status: `active`

Second-opinion PS5 emu (C#). Re-implement in Rust; do not vendor C#.

| Area | SharpEmu locus (approx.) | Target | Status | Commit | Notes |
|------|--------------------------|--------|--------|--------|-------|
| Out-of-process guest runner | Direct-execution worker isolation | `raeen-gui` | `partial` | `(working tree)` | Production Shell launches `raeen --run-eboot` as a child in a Windows kill-on-close Job Object. A hard-abort acceptance test proves the parent survives; native XInput/DualSense polling runs in the child. Structured bidirectional IPC, frame sharing, and the full crash-report schema remain open. |
| Zero-fault HLE leaf imports | `DirectExecutionBackend.Imports.cs` | `raeen-runtime` | `partial` | `(working tree)` | Compact eight-byte slots feed a generated SysV bridge for a reviewed leaf allow-list; context-changing imports retain VEH. The bridge switches to a private 256 KiB host stack and forwards six register args, eight stack args, and XMM0â€“7. One-million-call `strlen`: VEH 4.024 s / 248,480 calls/s versus direct 0.626 s / 1,598,025 calls/s (6.4Ã—; zero direct-path AVs). |
| LoadStartModule / module handles | KernelRuntimeCompatExports | `raeen-hle` / firmware | `done` | `fbac0b7` | pseudo-handle approach |
| libc string/mem + atexit | compat exports | `raeen-hle` | `done` | `c485704` | +review fixes `2ab04fa` (strstr DoS→memchr, truncation warn) |
| libc atoi/strtol/strtoul | compat exports | `raeen-hle` | `done` | `4f86ea0` | real base-0 parse + endptr |
| time / usleep | kernel time exports | `raeen-hle` | `done` | `922d0bf` | real host clock; usleep really sleeps |
| GetCompiledSdkVersion / getpid | KernelExports | `raeen-hle` | `done` | `9457258` | PS5 SDK 9.00 (Gen5); stable pid |
| SELF / eboot loader: PT_SCE_PROCPARAM capture + GetProcParam wiring | Loader | `raeen-firmware`/`runtime`/`kernel`/`hle` | `done` | `1787534`,`71221c5` | capture (SprxModule.procparam + proc_param_sdk_version); NOW wired end-to-end: LinkedModule.procparam_offset → runtime sets kernel.set_proc_param_addr(base+off) at load → sceKernelGetProcParam returns the real guest address (non-null sentinel only when the module has none). E2E tested |
| SELF / eboot loader: remaining (rebase, full param decode) | Loader | `raeen-firmware` | `todo` | | non-zero load bias rebase, PT_SCE_MODULE_PARAM name extraction |
| Sysmodule load chain (system modules, all HLE) | Libs | `raeen-hle` | `done` | `e2e9eb2` | sceSysmoduleLoadModule/LoadModuleInternalWithArg/Unload/IsLoaded all succeed — every SCE_SYSMODULE_* is HLE-registered, so "loading" resolves without bringing anything into memory. Panic-safe arg access + test |
| Real user .prx load chain (file-backed) | Kernel / Libs | `raeen-firmware`/`hle` | `todo` | | a title's own split-out .prx from disk — needs a module-load service reachable from HleContext (the M1-D architectural blocker) |
| Np (PSN) manager — offline | Np | `raeen-hle` | `done` | `df2ad63` | new libsce_np.rs: sceNpGetState reports SIGNED_OUT (SharpEmu value 1), reachability UNAVAILABLE, GetOnlineId="Player", callbacks accepted-but-never-fire — a title sees "offline", disables online features, and boots instead of hanging on a PSN check. Online Np (trophy sync, matchmaking) out of scope. 2 tests |
| Fiber / AMPR | Kernel | `raeen-hle`/`runtime` | `todo` | | verified blocked on infrastructure, not stubbable: sceFiberRun/Switch transfer control to *run the fiber's guest entry* → needs the guest execution-context machinery (M1-E family) — an init-only stub is the "no-op instead of real execution" anti-pattern. AMPR command buffers ultimately need a DMA/compute processing backend (overlaps M2). |
| Mouse / Ime / GameUpdate / SystemGesture / Share / NpGameIntent (headless) | Libs | `raeen-hle` | `done` | `3de1121`,`410e985` | libsce_peripheral.rs: libSceMouse (Open/Close/Read → 0 entries), libSceIme (Update/Keyboard* → no pending events), libSceGameUpdate (Init/Terminate OK), libSceSystemGesture (8 NIDs — recognizers OK, GetTouchEventsCount → 0), libSceShare (Init/SetContentParam OK), libSceNpGameIntent (Init OK). Each reports its benign headless "nothing here" state, resolving NIDs real titles poll (Quake KEX calls `sceImeUpdate` each frame) that were unresolved → crash. SharpEmu-cross-checked. 1 test. |
| Host input — native XInput + raw-HID DualSense readers | Host/Windows (`WindowsXInputReader.cs`, `WindowsDualSenseReader.cs`, `WindowsHidNative.cs`, `HostGamepadState.cs`) | `raeen-input` / `raeen-gui` | `done` | `(this session)` | new `xinput.rs` (pure `translate(XPadRaw)`→ControllerState: A→Cross/B→Circle/X→Square/Y→Triangle, shoulders→L1/R1, thumbs→L3/R3, Start→Options, Back→TouchPad, triggers→analog L2/R2, sticks (v+32768)>>8 w/ Y inverted), `hid.rs` (pure `parse_report` for USB 0x01 / BT 0x31 at fixed offsets via `windows-sys` SetupDi enumerate + CreateFileW/ReadFile — zero new deps; DualSense VID 0x054C, PID 0x0CE6/0x0DF2), and `native.rs` (`NativeGamepads` facade, two background threads behind `Mutex`, DualSense preferred). Merged into `shell/mod.rs::push_pad_state` ahead of gilrs→keyboard, fixing the all-zeros-UUID gilrs "No mapping found" dead-button case (Steam Input / DS4Windows / generic HID). `#[cfg(windows)]`-gated; non-Windows no-ops. **Input only** — rumble/lightbar output reports deliberately deferred. 10 tests (5 XInput + 5 DualSense). |
| Media handshakes — Ajm / Ngs2 / AvPlayer / Ult | Libs | `raeen-hle` | `done` | `143f741` | new libsce_media.rs: faithful port of SharpEmu's Ajm/Ngs2/AvPlayer/Ult (themselves handshake stubs — no real DSP in the reference). Ajm init (context id), Ngs2 system/rack/voice-handle mgmt + idle VoiceGetState, AvPlayer Init(null handle)/IsActive(inactive)/GetVideo/AudioData(no frame)/Close, Ult init. Subsystems initialize so titles proceed; **no real media output yet** (same shape as AudioOut/VideoOut). Real decode/synthesis = follow-up. 3 tests. |
| libSceAjm batch silence stubs (Bink/AJM hot path) | AjmExports.cs `2272b9b`,`d3600c9` | `raeen-hle` | `done` | `(this session)` | libsce_media.rs: `sceAjmBatchJobDecode` is now a real **silence** stub — best-effort advance of the AjmBatchInfo cursor (one 64-byte job run), zero the guest PCM out (bounded to 1 MiB), and write the 32-byte decode sideband (input reported fully consumed, 1 frame) so the title advances its bitstream instead of spinning. `sceAjmBatchInitialize`→OK; added `sceAjmBatchStartBuffer` (6-arg) and aligned `sceAjmBatchStart` to SharpEmu's 5-arg shape (clear the AjmBatchError struct, publish a batch id); `sceAjmBatchWait` clears the error struct. **No per-call WARN**; accepts Gen5 codec instance ids (no codec validation on the decode path). 3 tests. |
| libSceAgc `*GetSize` packet-sizing probes | AgcExports.cs `74a5198` | `raeen-hle` | `done` | `(this session)` | libsce_agc.rs: registered the 6 missing sizing NIDs — `sceAgc{Dcb,Acb}DmaDataGetSize`, `sceAgcDcbDrawIndexIndirectGetSize`, `sceAgcDcbSetIndexCountGetSize`, `sceAgcDcbStallCommandBufferParserGetSize`, `sceAgcDcbGetLodStatsGetSize` — each returning the per-packet byte size in rax **only** (no guest-memory writes). Export names hash-verified against SharpEmu's declared NIDs; sizes cross-checked against the writers in the same file. These were NOT_FOUND at RenderThread startup → null packet pointer → immediate write AV before any GPU work. 1 test. |
| Gen5 `sceAgcCreateShader` program-register discovery | AgcExports.cs `8e1e89c` | `raeen-hle` | `done` | `(this session)` | Ported the generic Gen5 fix: search the complete SH table for compute/PS/VS/ES/GS/HS/LS PGM_LO/HI pairs instead of assuming entries 0/1. Type-5 hull headers beginning with HS RSRC1/RSRC2 and omitting PGM_LO/HI are accepted because direct register commands publish the code address later. Focused tests: 2/2. GTA V PPSA04264 01.005.000 reaches 342 CreateShader calls without a bind failure, but still does not submit/present: the next measured worker assertion follows failure to locate `common:/shaders/fxdb/sga_prospero_final_init.awc`; do not report GTA as rendering. |
| VFS guest→host path sandbox (drive-letter / symlink escape) | KernelMemoryCompatExports.cs `ResolveGuestPath`/`CombineWithinMount`/`EscapesMountViaReparsePoint` `e01092a` (tests `KernelSandboxEscapeTests.cs`) | `raeen-kernel` | `done` | `(this session)` | `filesystem::resolve_path` now delegates to a new `combine_within_mount`: sanitize each relative segment (reject `..`, an absolute segment, a `:`-qualified drive/ADS token, NUL bytes, and over-long names), walk the existing components refusing symlinks/reparse points, canonicalize the deepest existing ancestor and **assert** it stays under the canonicalized mount root, and **fail closed** on any error. Closes the Windows drive-letter escape (`/app0/C:/Windows/...` where `Path::join` replaced the base → arbitrary host read/write/delete) and the symlink/junction escape. 6 new tests (drive-letter, traversal, NUL, over-long, legit nested, reparse). |
| Json (`sce::Json` C++ lifecycle) + VoiceQoS | Libs | `raeen-hle` | `done` | `10dfa92` | new libsce_json.rs: port of SharpEmu's `sce::Json` exports — mangled MemAllocator/Initializer/InitParameter2 ctors (return `this`), dtors (return 0), fluent `setAllocator`/`setFileBufferSize` (return `this`), `Initializer::initialize` (OK / EINVAL for null this); registered under `libSceJson`+`libSceJson2`. C++-ABI-correct lifecycle (parsing itself not exercised by these NIDs). Plus `libSceVoiceQoS` `sceVoiceQoSInit` (no voice backend). 3 tests. |
| Rtc — calendar math + clock (complete) | Libs | `raeen-hle` | `done` | `726ab2c`,`bd9d79e`,`5407aa7` | new libsce_rtc.rs: port of SharpEmu RtcExports. 25 NIDs — GetTickResolution/IsLeapYear/GetDaysInMonth/GetDayOfWeek (Sakamoto, Sun=0), CheckValid (field-by-field `SceRtcDateTime` validation w/ Orbis error codes), TickAdd{Ticks/µs/Sec/Min/Hour/Day/Week} (signed µs tick arithmetic, over/underflow→error), CompareTick(-1/0/1), GetCurrentTick + Network/RawNetwork/AdNetwork/DebugNetwork aliases (host UTC as an Rtc tick), **GetCurrentClock/LocalTime** (tick→`SceRtcDateTime` via Hinnant civil-from-days). Tick = µs since 0001-01-01. 7 tests. **libSceRtc complete.** |
| PlayGo | Libs | `raeen-hle` | `done` | `39f7e58` | new libsce_playgo.rs: all chunks LOCUS_LOCAL_FAST, progress==total (complete), empty to-do list, handle out — "everything installed" so titles skip download gating (SharpEmu-cross-checked values). 13 NIDs; 3 tests |
| UserService (initial user / login list / name / event) | UserService | `raeen-hle` | `done` | `3d44b94`, `(this session)` | libsce_user_service.rs: retail-style local user id `0x10000000`, GetInitialUser/GetLoginUserIdList/GetUserName("Player"), and a process-scoped one-shot login event followed by `NO_EVENT`. The login transition is cross-checked against current SharpEmu, KytyPS5, and shadPS4 and is what lets Minecraft enter its Pad path. 6 NIDs; 5 focused tests |
| SystemService (status / param / safe-area) | SystemService | `raeen-hle` | `done` | `46432e4` | new libsce_system_service.rs: GetStatus(eventNum=0, quiet), ParamGetInt (SharpEmu mapping 1/2/3/1000→1, 4→180), GetDisplaySafeAreaInfo(ratio 1.0), HideSplashScreen/ReportAbnormalTermination OK — a title's per-frame poll runs undisturbed. 5 NIDs; 3 tests |
| pthread **mutexes** (state machine) | Kernel | `raeen-hle` | `done` | `cf265b6` | `pthread_sync.rs` — faithful port of SharpEmu's mutex state machine (type/owner/recursion) into per-process kernel state (`OrbisKernel::pthread_mutexes`/`pthread_mutex_attrs`, keyed by both `pthread_mutex_t` addr and allocated handle). Init writes a real opaque handle; Lock/Trylock/Unlock honor recursive (count) / error-check (`EDEADLK`/`EBUSY` on self-relock) / normal-adaptive (lenient) semantics; Destroy + mutexattr Init/Settype/Destroy. Correct for single-active-execution (1 guest thread → ownership = recursion + type). 5 tests. Adds Trylock/Destroy/mutexattr* (new NIDs). Replaces the return-0 stubs. |
| **BSD sockets** (offline) + net helpers (`socket`/`bind`/`connect`/`getsockname`/`htons`/`inet_pton`/`bzero`) | Kernel | `raeen-hle` | `done` | `574447f` | `kernel_socket.rs` — port of SharpEmu's `KernelSocketCompatExports` **minus its real host-TCP path** (Raeen models no host connectivity; guest code must never reach the host network). socket/bind/getsockname track offline state (`OrbisKernel::kernel_sockets`, fds in a high range); connect always fails. Pure helpers fully correct: htons (byte swap), inet_pton (dotted-quad→4 bytes), bzero (zero guest mem). 4 tests. Security note: deliberately no host TCP. |
| **Kernel event queues** + user events (`sceKernelCreateEqueue`/`AddUserEvent`/`TriggerUserEvent`/`WaitEqueue`/`GetEvent*`) | Kernel | `raeen-hle` | `done` | `e2b3432` | `kernel_equeue.rs` — port of the user-event core of SharpEmu's `KernelEventQueueCompatExports` into `OrbisKernel::kernel_equeues`/`kernel_equeue_events`. Replaces libkernel's fake-handle-1 stub. Create/Delete, AddUserEvent[Edge]/DeleteUserEvent, TriggerUserEvent (pending+udata), WaitEqueue (delivers 32-byte `SceKernelEvent` structs — ident/filter=EVFILT_USER/fflags/udata — edge-clears, writes count; ETIMEDOUT if none), GetEventId/Filter/Data/UserData. Registration/trigger/delivery **fully correct**; true blocking wait + AMPR/graphics events need M1-E/M2. 3 tests. |
| **Kernel semaphores** (`sceKernelCreate/Signal/Wait/Poll/Cancel/DeleteSema`) | Kernel | `raeen-hle` | `done` | `53c78ee` | `kernel_semaphore.rs` — port of SharpEmu's `KernelSemaphoreCompatExports` into `OrbisKernel::kernel_semaphores` (count + max per handle). Create (validation, count=init, u32 handle out), Signal (+count, error if > max), Poll (dec if available else EBUSY), Wait (dec if available else ETIMEDOUT), Cancel (reset + 0 waiters), Delete. Count arithmetic **fully correct**; true blocking Wait needs the M1-E scheduler. SCE error codes. 5 tests. Previously-unimplemented NIDs. |
| **Kernel event flags** (`sceKernelCreate/Set/Clear/Poll/Wait/Cancel/DeleteEventFlag`) | Kernel | `raeen-hle` | `done` | `92b1fdf` | `kernel_eventflag.rs` — port of SharpEmu's `KernelEventFlagCompatExports` into `OrbisKernel::kernel_event_flags` (64-bit condition bits per handle). Create (attr/name validation, initial bits, handle out), Set (OR), Clear (`&= pattern`, Orbis semantics), Poll (AND/OR + CLEAR_ALL/PATTERN, EBUSY if unmet), Wait (OK if satisfied else ETIMEDOUT), Cancel (force bits + 0 waiters), Delete. Bit state **fully correct**; true blocking Wait needs the M1-E scheduler. SCE error codes. 5 tests. Previously-unimplemented NIDs. |
| pthread **thread identity/control** (`Self`/`Equal`/`Yield`/`Rename`/`Getthreadid`) | Kernel | `raeen-hle` | `done` | `a3e1170` | `pthread_thread.rs` — the small stateless pthread calls a title makes constantly (SharpEmu-cross-checked). `Self`/`Getthreadid` return the one guest thread handle, `Equal` compares, `Yield` succeeds (nothing to switch to), `Rename` accepts + logs. **Complete and exact** for single-active-execution. 3 tests. All previously-unimplemented NIDs (were unresolved → crash). |
| pthread **TLS keys** (`scePthreadKey*`/`Set`/`Getspecific`) | Kernel | `raeen-hle` | `done` | `d1f4e1b` | `pthread_tls.rs` — port of SharpEmu's pthread-key TLS into `OrbisKernel::pthread_tls_keys`/`pthread_tls_values`. KeyCreate allocates a monotonic key (+ destructor) and writes it to `*key`; Setspecific stores a per-thread value (EINVAL for unknown key); Getspecific returns it (0 if unset/unknown, value IS the return per ABI); KeyDelete drops the key + all its values. **Fully complete** — the (thread, key) map is exact under single-active-execution, no runtime dependency. libc/runtimes use this for thread-local storage. 4 tests. All previously-unimplemented NIDs. |
| pthread **thread attributes** (`scePthreadAttr*`) | Kernel | `raeen-hle` | `done` | `7908c41` | `pthread_attr.rs` — port of SharpEmu's `PthreadAttrState` into `OrbisKernel::pthread_attrs` (detach/stack/guard/sched, keyed by addr + handle). Init writes a handle + default state (1 MiB stack, joinable); Set/Get detachstate/stacksize/guardsize + Set schedpolicy round-trip (Get writes to the guest out-ptr); Destroy clears. **Fully complete** — thread attributes are pure config with no runtime dependency (unlike thread *creation*). 3 tests. All previously-unimplemented NIDs. |
| pthread **rwlocks** (state machine) | Kernel | `raeen-hle` | `done` | `4883277` | `pthread_sync.rs` — port of SharpEmu's `PthreadRwlockState` into `OrbisKernel::pthread_rwlocks` (readers / writer / writer_recursion, keyed by addr + handle). Init writes a real handle; Rdlock/Tryrdlock nest read holds, Wrlock/Trywrlock acquire-or-recurse the write hold, Unlock releases write-then-read (`EPERM` if neither held); Destroy + rwlockattr Init/Destroy. Correct for single-active-execution. 3 tests. New NIDs (was fully unimplemented). |
| pthread threads / create-join / cond blocking | Kernel | `raeen-runtime`/`hle` | `todo` | | M1-E — `scePthreadCreate` running the thread entry + `CondWait`/blocking need the per-thread guest scheduler (a real 2nd guest context); an init-only stub is the "no-op instead of real execution" anti-pattern. Mutex + rwlock *state* is now done (above). |
| AudioOut (pacing stub, no playback yet) | Audio | `raeen-hle` | `done` | `bedb5e0` | libsce_audio_out.rs: Init/Open(handle+grain/freq)/Output(acks buffer, sleeps ~grain÷freq bounded, returns sample count)/Close/SetVolume — audio thread paces without hang or 100% spin (M3 "audio must not hang"). Real host playback (cpal) is the follow-up. 3 tests |
| VideoOut: flip-completion + resolution (no real present yet) | Graphics | `raeen-hle` | `done` | `f908ddb` | SubmitFlip bumps a flip counter + records flipArg; GetFlipStatus reports count + zero-pending so the render loop advances (was stalling); GetResolutionStatus=1080p, GetVblankStatus=frame counter. Real swapchain present = M2/M3 follow-up. 2 tests |
| VideoOut real present / AGC / shaders→Vulkan | Graphics | `raeen-gpu`/`hle` | `wip` | synthetic M2 proof | AGC PM4 → kyty SPIR-V → offscreen is proven only by a synthetic fixture; M2 remains open. GraphicsRun CP now survives retail DCBs (skip-unknown policy) and handles Gen5 register/index/indirect ops — see the `GraphicsRun.cpp` CommandProcessor row. Remaining: VideoOut → swapchain, GraphicsRender (real indexed draws, guest shader bind), and non-synthetic acceptance evidence. |
| DualSense / pad (digital+analog state) | Input | `raeen-input`/`hle` | `done` | `0ceb7db` | ControllerState→Orbis ScePadData encoder (documented button masks, stick/trigger byte mapping) in raeen-input; scePadReadState writes a valid state + returns 1 (was garbage + 0 → homebrew read-loop hang). Live host-input routing (InputManager→HleContext) + haptics/adaptive-triggers still todo |
| DualSense: live input routing (kernel snapshot → scePadReadState) | Input | `raeen-kernel`/`hle` | `done` | `cb1b56d` | kernel holds a settable 12-byte pad-state snapshot (OrbisKernel::set_pad_state/pad_state); scePadReadState reads it (neutral fallback when unset) — live host input now flows guest-ward. Remaining: Shell polling InputManager into set_pad_state each frame (UI wiring) + haptics/adaptive triggers |
| Filesystem: open/read/close/lseek | KernelExports/FS | `raeen-kernel`/`hle` | `done` | `896495d` | VFS-backed, real host files under /app0; write persistence + fstat still todo |
| Filesystem: write persistence (savedata) | FS | `raeen-kernel`/`hle` | `done` | `5285857` | VFS honors O_WRONLY/RDWR/CREAT/TRUNC/APPEND; write buffers + flush-on-close to host file; ".." traversal refused on writable open; hle write() routes non-console fds to VFS; hle open() honors O_CREAT. E2E: guest open+write+close persists to host, read-back works |
| Filesystem: fstat / directory ops | FS | `raeen-kernel`/`hle` | `todo` | | needs SCE stat struct layout |
| SaveData mount | SaveData | `raeen-hle` | `done` | `93a7c0a` | new libsce_save_data.rs (was empty stub): sceSaveDataMount{,2,3} writes the /savedata0 mount point into the 64-byte result; Umount/Initialize/Terminate OK. Completes the save path — mount → open/write under /savedata0 → VFS persists to host savedata dir (write-persistence 5285857). SharpEmu-cross-checked. 2 tests |
| CommonDialog / MsgDialog | CommonDialog | `raeen-hle` | `done` | `264d895` | new libsce_common_dialog.rs: no host popup → sceMsgDialogOpen completes immediately (status→FINISHED), GetResult reports the OK button, so a title's dialog poll loop finishes instead of hanging. Status None/Init/Finished per SharpEmu. 2 tests |
| GUI patterns | app | `raeen-gui` | `skip` | | optional UX only |
| NpTrophy2 (Gen5 trophy ctx/handle lifecycle) | Np | `raeen-hle` | `done` | (uncommitted) | new `libsce_np_trophy2.rs`: faithful port of `NpTrophy2Exports`. CreateContext/CreateHandle write monotonic int32 ids (no id consumed on fault), Destroy/Abort/Register/(Un)RegisterUnlockCallback/ShowTrophyList → OK. Honest handshake stub — no trophy is ever unlocked/displayed. 9 NIDs; 1 test |
| NpUniversalDataSystem (UDS telemetry) | Np | `raeen-hle` | `done` | (uncommitted) | new `libsce_np_universal_data.rs`: port of `NpUniversalDataSystemExports`. Initialize validates+reads 16-byte param (lib code 0x80553102 on null), CreateContext writes id 1 (null out = benign OK), CreateHandle writes monotonic handle to first writable of Rdi/Rsi, Register/DestroyHandle → OK. No event transmitted. 5 NIDs; 3 tests |
| NpWebApi2 (PSN REST client init/term) | Np | `raeen-hle` | `done` | (uncommitted) | new `libsce_np_web_api2.rs`: port of `NpWebApi2Exports`. Initialize validates (httpCtxId>0, poolSize≠0; lib code 0x80553402) + sets initialized flag, ForToolkit accepts unconditionally, Terminate clears. No HTTP request issued. 3 NIDs; 1 test |
| NpEntitlementAccess (DLC entitlement query) | Np | `raeen-hle` | `done` | (uncommitted) | new `libsce_np_entitlement.rs`: port of `NpEntitlementAccessExports`. Initialize zeroes the 0x20-byte boot-param block; GetAddcontEntitlementInfoList writes an empty (zeroed) 0x10-byte list header — title sees "no DLC owned". 2 NIDs; 2 tests |
| NpSessionSignaling (P2P signaling init) | Np | `raeen-hle` | `done` | (uncommitted) | new `libsce_np_session_signaling.rs`: port of `NpSessionSignalingExports`. The sole Initialize is an honest no-op → OK; no peer connection established (no net backend). 1 NID; 1 test |
| NpManagerForToolkit (state callback) | Np | `raeen-hle` | `done` | (uncommitted) | added to `libsce_np.rs`: `sceNpRegisterStateCallbackForToolkit` under the sibling `libSceNpManagerForToolkit` library, sharing the offline Np-state callback handler. 1 NID; 1 test |
| Http / Http2 (HTTP-client ctx/template lifecycle) | Network | `raeen-hle` | `done` | (uncommitted) | new `libsce_http.rs`: port of `HttpExports`+`Http2Exports`. Init allocates+records monotonic context ids (returned in rax; 0-pool → lib codes 0x804311FE/0x80436016), CreateTemplate validates the context (0x80431100) + allocates a 0x1001+ template id, Term cascade-removes a context's templates, Delete/Term validate ids. Registries in `OrbisKernel` (per process). **No host network — no transfer ever sent.** 6 NIDs; 2 tests |
| Ssl (SSL/TLS context lifecycle) | Network | `raeen-hle` | `done` | (uncommitted) | new `libsce_ssl.rs`: port of `SslExports`. Init allocates+records a monotonic context id (0-pool → 0x8095F008), Term validates+removes (0x8095F006), Close is unconditional OK. Registry in `OrbisKernel`. **No TLS handshake performed (no net backend).** 3 NIDs; 1 test |
| AudioOut2 (Gen5 audio ctx/port lifecycle) | Audio | `raeen-hle` | `done` | (uncommitted) | new `libsce_audio_out2.rs`: port of `AudioOut2Exports` (distinct from older libSceAudioOut). ResetParam fills the 0x30-byte param (2ch/48kHz/0x400 grain, prefix-only per SharpEmu's Quake-canary note), QueryMemory→0x10000, Context/User/Port Create hand back opaque handles (port handle encodes type + rolling 8-bit id), PortGetState/GetSpeakerInfo fill their structs, Destroy→OK. **PCM now plays: `ContextPush` → `pcm.rs` s16/float→stereo convert → `raeen_audio::output::submit` (cpal), matching libSceAudioOut.** 11 NIDs; 4 tests |
| Coredump (crash-handler registration) | Kernel | `raeen-hle` | `done` | (uncommitted) | new `libsce_coredump.rs`: port of `sceCoredumpRegisterCoredumpHandler`. Records handler ptr + user context; never invoked (no crash-path guest callback yet). 1 NID; 1 test |
| ShareUtility (content-param handshake) | Share | `raeen-hle` | `done` | (uncommitted) | new `libsce_share.rs`: port of `ShareExports` under SharpEmu's **`libSceShareUtility`** LibraryName. Initialize validates (0 mem → EINVAL) + sets flag; SetContentParam reads+retains the NUL-terminated UTF-8 string (EFAULT if unreadable/unterminated). Supersedes the mis-named `libSceShare` stub in `libsce_peripheral.rs`. 2 NIDs; 3 tests |
| libScePosix (POSIX pthread_exit alias) | Kernel | `raeen-hle` | `done` | (uncommitted) | added to `libkernel.rs`: SharpEmu's sole `libScePosix` export `pthread_exit` is aliased onto the existing `scePthreadExit` handler (no duplicated logic). 1 NID |

**Note (still graphics-gated):** `libSceAgcDriver` register-defaults tables + `Agc/`/`Ampr/` command-*content* writers remain deferred to the Vulkan/compute backend (see the Kyty `emulator/Libs` row above) — unchanged by this batch.

**Delete SharpEmu when:** all non-`skip` rows `done`, and no open M# work still citing this tree.

---

## KytyPS5 (`reference/kytyps5`) — status: optional

| Area | Target | Status | Notes |
|------|--------|--------|-------|
| PS5 deltas over Kyty (SRT, pthread, VM, LibUlt, …) | merge into kyty-* / raeen-* | `todo` | study; don’t blind-merge |
| Commercial boot paths | docs + HLE/GPU | `todo` | |

---

## shadPS4 (`reference/shadps4`) — status: optional

Pattern reference for Orbis HLE (memory, libkernel, linker). Port selectively; no need to 1:1 the whole tree.

| Area | Target | Status | Notes |
|------|--------|--------|-------|
| Memory model / libkernel patterns | `raeen-hle`/`kernel` | `todo` | |
| Linker / NID ideas | `raeen-firmware` | `todo` | |
| Vulkan present path ideas | `raeen-gpu` | `todo` | |

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
| 2026-07-23 | Audited SharpEmu through `6db095e` (no newer `origin/main` delta). Avatar probe chain added process-backed `getargc/getargv`, `sceKernelMtypeprotect`, PS5 PlayGo install/optional-chunk queries, and KytyPS5 `sceAgcGetPacketSize`. Also fixed PlayGo's `uint32_t *outEntries` overwrite (Raeen incorrectly wrote 8 bytes). Measured progress: 21 HLE calls at the original stop → 4,096 calls, multithreaded boot, and first AGC submissions. |
| 2026-07-23 | Ported SharpEmu `8e1e89c` Gen5 hull-header handling and added opt-in `RAEEN_DIAG_STAT_MISS` path diagnostics. GTA V probe: 2,733 HLE calls and 20 guest workers before `sga_prospero_final_init` asserts; no AGC submit/draw/flip yet. SharpEmu PR #454 likewise says only “getting further into loading,” not GTA gameplay/rendering. |
| 2026-07-23 | Refreshed all four graphics references: SharpEmu `21f964a` (delta from `6db095e` is README-only; no code to port), Kyty `4733b7e` (no material graphics delta), KytyPS5 `8587638` (GPU tiler now dispatches 2D 8×8 workgroups; queued for the GPU-detiling performance slice), and shadPS4 `8161049` (priority-pending-op mutex fix plus sparse guest-memory copies at `cca6405`/`dee3a04`). Raeen's submission lifecycle already serializes its pending queue; sparse-copy applicability remains under guest-memory audit rather than being copied blindly. |
| 2026-07-23 | Replaced Minecraft's loud `-KRzWekV120` AGC stub with KytyPS5's measured `GraphicsUnknownKRzWekV120` three-DWORD emitter (`c0017a00 20000243 control`), including exact packet/cursor tests. Added the missing `IT_SET_UCONFIG_REG_INDEX` consumer so its `VGT_INDEX_TYPE` write reaches indexed draws instead of being skipped as unknown opcode `0x7a`. SharpEmu still exposes this NID as a return-zero stub; KytyPS5 supplies the useful command semantics. |
| 2026-07-23 | Applied Kyty's `ShaderGetBindIds` structural-cache principle to Raeen's active SPIR-V cache, extended for Raeen's Gen5 texture dimension/format, embedded constants, EUD/global-memory and LDS codegen. Runtime guest addresses and descriptor payloads no longer churn modules; fresh analyzed metadata still drives every bind. Minecraft A/B fell from >2,100 compile events to 81/8 addresses with present 256 and no failures. Also removed Kyty's PS4-era writable-pixel-buffer restriction only for Gen5, because Raeen's existing graphics descriptor path already carries fragment-visible storage buffers. Astro A/B reached 6 presents with zero translation/analysis errors. Fragment SSBO guest-memory writeback remains open; no M2/M3 closure claim. |
| 2026-07-24 | Ported Kyty's explicit Gen5 stencil-operation conversion into `raeen-gpu` instead of casting the AMD opcode to Vulkan. This preserves AMD Ones/ReplaceTest/ReplaceOp, clamp/invert/wrap, and the distinct test/operation reference values; the old cast caused Minecraft's stencil-tested UI to rasterize no fragments. Added persistent-stencil regression coverage. Also added original content-aware scanout selection, black-frame/pending-composite diagnostics, bounded indexed vertex uploads, and correlated `Fetch*`/POS0 tracing. A no-bypass release probe produced a real Minecraft loading frame and then visible exact scanouts; this is not a gameplay or milestone-closure claim. |
| 2026-07-24 | Replaced UserService's permanent `NO_EVENT` with a process-scoped initial Login event and moved the primary local user to the retail-style `0x10000000` id used by current SharpEmu; KytyPS5 and shadPS4 independently confirm the login-event transition. A release Minecraft run then consumed the event, opened Pad handle 1, and read a live raw-HID DualSense snapshot. The same run preserved the complete title panorama at present 2048. Button-driven transition/gameplay acceptance remains open. |
