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

### Remaining-work classification (as of the latest batch)

Every **self-contained, verifiable** module in both references is `done`/`skip`
— including *every* SharpEmu-implemented libSce. What keeps both trees `active`
(not yet `fully_ported`) is exclusively work that needs a **subsystem stood up**
or a **real fixture**, none of which can be honestly completed as a stub:

1. **M2 GPU draw pipeline** — Kyty `emulator/Graphics` (PM4→GCN-shader→Vulkan) +
   SharpEmu VideoOut real present + GraphicsDriver + macro/other tiling modes.
   Needs the draw path built and verified against real command streams / rendered
   output. (Tiling Z-order + flip-status already done.)
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
| VirtualMemory (Core wrapper) | `kyty-core` | `done` | `e06b0d3` | `virtual_memory.rs` forwards 1:1 to sys_virtual (as Kyty's Core does on Windows); ExceptionHandler `skip` — xps5x-runtime VEH supersedes |
| MemoryAlloc / MSpace | `kyty-core` | `skip` | | manual C++ heap (`mem_alloc`/`mem_free` + stats) — superseded by Rust's global allocator (host) + `xps5x-runtime` `GuestArena` (guest); same rationale as skipped `SafeDelete`. Convention: manual-memory scaffolding → safe Rust equivalent. |
| Threads | `kyty-core`/`xps5x-runtime` | `todo` | | overlaps M1-E (real guest threads) — deferred, see SDD sketch |
| File | `kyty-core` | `skip` | | 1311-line buffered File class over sys_file_io — superseded by xps5x-kernel VFS on the hot path; verified the one ported consumer (json_reader) uses std, not Core::File, so nothing needs it. Port later only if a future Kyty subsystem does |
| SDLSubsystem | `kyty-core` | `skip` | | SDL window/input/audio init — superseded by xps5x-gui's eframe/egui (verified main.rs uses eframe) + xps5x-input/audio crates |
| Debug / Subsystems / Core.cpp | `kyty-core` | `skip` | | verified: Subsystems=a dependency-ordered init/shutdown manager (superseded by XPS5X's per-crate `new()` init, no central manager); Debug/DebugMap=C++ symbol-map (.map/.csv/MSVC) loading for backtrace symbolication (superseded by Rust's native backtraces + `tracing`); Core.cpp=Core::Init glue. All init-glue/scaffolding XPS5X's architecture replaces. sys_dbg (the substantive Sys-layer piece) already ported. |

### Later Kyty trees

| Module | Target | Status | Commit | Notes |
|--------|--------|--------|--------|-------|
| lib/Math: VectorAndMatrix (Vec2/3/4, Mat2/3/4) | `kyty-math` | `done` | `f9ecddf` | `vector_and_matrix.rs` aliases Kyty vec/mat to `glam` (column-major, GNM/GLSL-order) + Kyty-named ctor helpers (splat/vec3_w/identity); 4 tests |
| lib/Math: Rand (mt19937) | `kyty-math` | `done` | `f9ecddf` | `rand.rs` — Kyty `Rand::*` API (uint/int/double/float + inclusive/exclusive ranges + seed) over the `rand` crate (StdRng, thread-local; not bit-identical to mt19937 — clean-room, sequence not load-bearing); 4 tests |
| lib/Math: Crypto (AES + Hash) | `kyty-math` | `skip` | | AES/SHA → RustCrypto (`aes`/`cbc`/`sha1` already workspace deps used by xps5x-firmware SELF decrypt); 3rdparty→workspace-crate convention, do not transliterate |
| lib/Scripts | `kyty-scripts` | `skip` | | Lua scripting — unused by XPS5X's execution path (guest games are native binaries, not Kyty Lua demos); per goal "skip unused Scripts/lua unless config needs it" |
| emulator/Loader | → `xps5x-firmware`/`xps5x-loader` | `wip` | (M0/M1) | realized on the hot path: SELF decrypt-or-passthrough → sprx parse → PT_SCE_DYNLIBDATA/PT_DYNAMIC decode → NID link (LM0/LM1), + PT_TLS + PT_SCE_PROCPARAM capture. Remaining: non-zero load-bias rebase, real user .prx chain, full param decode — see loader `todo` rows above |
| emulator/Kernel | → `xps5x-kernel`/`xps5x-hle` | `wip` | (M1) | realized: OrbisKernel (VMM, VFS, thread mgr, module table, console, proc-param, pad state) + libkernel HLE (mem/time/module/proc-time/fd/file I/O). Remaining: real threading (M1-E), Fiber/AMPR, broader syscall surface |
| emulator/Libs (libSce* HLE surface) | `xps5x-hle` | `wip` | (many) | realized as xps5x-hle libSce* modules: libc, libkernel (mem/time/module/proc-time), Sysmodule, PlayGo, User/System Service, Pad, AudioOut, VideoOut, SaveData, CommonDialog/MsgDialog, AppContent, Np, Net/NetCtl, DiscMap, **Rtc**. The high-traffic boot/HUD libs are done, plus Rtc + Mouse/Ime/GameUpdate; the media-subsystem handshakes (Ajm/Ngs2/AvPlayer/Ult) are now ported too (as handshake stubs, matching SharpEmu — no real output yet). **Every SharpEmu libSce that isn't GPU-command or scheduler infrastructure is now done.** The only remaining SharpEmu libSce are **Agc** (66 GPU-command-building exports) + **Ampr/Fiber** — these are the M2-GPU backend and M1-E scheduler infrastructure gates, not standalone-portable. Real audio/video *output* for AudioOut/Ngs2/Ajm/AvPlayer is a follow-up (host backend), not a port gap. GraphicsDriver = the M2 pipeline row. |
| emulator/Graphics: texture micro-tiling | `xps5x-gpu` | `done` | `09f44c0` | fixed detile_micro from a bogus linear (py*8+px) interior to the documented GCN thin micro-tile Z-order (Morton interleave x0y0x1y1x2y2); added inverse tile_micro + round-trip/bijection/known-mapping tests. DEPTH/DISPLAY/ROTATED modes + macro bank/pipe swizzle + hardware-exact validation vs real dumps still todo |
| emulator/Graphics: PM4 command decoder | `xps5x-gpu` | `wip` | `9cee071` | first verified slice of the pipeline: added tests for process_command_buffer (Type 0 reg write, Type 3 SET_CONTEXT_REG + DRAW/DISPATCH, Type 2 NOP) — decode + register state + stats asserted against hand-built PM4 buffers. Found an API smell (registers write_context takes an offset but read_context takes an absolute addr — bug-prone, flagged). Remaining: full opcode coverage, shader (GCN→SPIR-V), Vulkan submit, real present |
| emulator/Graphics: SPIR-V emitter (header/structure) | `xps5x-gpu` | `wip` | `ddedfab` | verified the SPIR-V emitter produces structurally-valid modules for every shader stage: correct magic (0x07230203) + version + 5-word header + nonzero id-bound, pixel shader emits OpExecutionMode(OriginUpperLeft), geometry shader declares the Geometry capability. Remaining: real IR→SPIR-V body emission (currently a minimal main{ret}), GCN decode→IR, Vulkan submit + present |
| emulator/Graphics: SPIR-V constant pool | `xps5x-gpu` | `wip` | `2f51472` | `emit_spirv` declares an i32 scalar type and materializes each distinct IR constant as an `OpConstant` (ints on i32, floats on f32), deduped by (type, bits) — the first real body-emission step past `main{ret}`, producing the ids the arithmetic body will reference. Verified by parsing the emitted module back: constant present, emitted once for a repeated value, absent when the program has none |
| emulator/Graphics: rspirv structural validation gate | `xps5x-gpu` | `done` | `df1f661` | added `rspirv` 0.12 as a **test-only** dev-dependency (MIT license option — GPL-2.0-only compatible, not linked into the emulator; attributed in `THIRD_PARTY_NOTICES.md` `42899be`). Every emitted module is now parsed via `rspirv::dr::load_words` — real structural validation (magic/version, per-instruction word counts, operand layout) across all six stages + a constant-pool module. User-approved dependency choice (rspirv over Apache-2.0 spirv-val). |
| emulator/Graphics: SPIR-V arithmetic body | `xps5x-gpu` | `wip` | `8ce7dc8` | `emit_spirv` lowers the IR body: walks nodes in SSA order, resolves sources to constant-pool ids / prior SSA result ids / type-correct shared `OpUndef` (unwired live-ins), and emits `OpFAdd/FSub/FMul/FDiv` + `OpIAdd/ISub/IMul` with fresh result ids threaded to later uses. Verified via rspirv's **parsed** module: `r0=2.0+3.0; r1=r0*r0` → an `OpFAdd` then an `OpFMul` referencing the add's result; whole module parses. Remaining: real I/O interface vars (`OpVariable`+`OpLoad`/`OpStore` replacing the undefs) and full `spirv-val` semantic validation, then Vulkan submit + present |
| emulator/Graphics: SPIR-V logical-layout + input interface vars | `xps5x-gpu` | `wip` | `30de312` | **(a) Layout correctness:** the builder emits into per-section buffers (caps → memory-model → entry-points → exec-modes → annotations → types/vars → functions) concatenated in SPIR-V's mandated order; previously types preceded `OpEntryPoint` (rspirv tolerated it, `spirv-val` would not). Test asserts `OpEntryPoint` precedes the first type decl. **(b) Inputs:** each distinct `IrValue::Input(loc)` becomes an Input-storage `OpVariable` (ptr-to-f32) with `OpDecorate Location`, listed in the entry-point interface and `OpLoad`-ed once (cached) in the body — replacing the input `OpUndef`s. Verified via rspirv's parsed module (2 inputs → 2 Input vars, both in interface, body OpLoads + OpFAdd). |
| emulator/Graphics: SPIR-V output interface vars | `xps5x-gpu` | `wip` | `d153bd0` | each export node (`ExportColor/Position/Param`) declares an Output-storage `OpVariable` (ptr-to-f32) at a successive Location, listed in the entry interface, and the body `OpStore`s the export's resolved value into it. **Shader I/O path now complete: inputs `OpLoad` → arithmetic → exports `OpStore`, in correct logical layout, rspirv-validated** (`r0=in0+in1; export r0` → 1 Output var in interface + an OpStore). Remaining: int/vec4 I/O types, position/param builtin decorations, full `spirv-val` semantic pass, then Vulkan submit + present |
| emulator/Graphics: GCN instruction decoder | `xps5x-gpu` | `wip` | `3acad0f` | verified the RDNA2 decoder: encoding classification (VOP2/VOP3/SOPP) + instruction widths (4/8B), 8-byte raw assembly (word1<<32\|word0), stream walk with byte-offset advance, and stop-at-S_ENDPGM. <4-byte binary errors. Remaining: full operand (src/dst) decode + the GCN→IR lowering that feeds the SPIR-V body |
| emulator/Graphics: GCN→IR lowering (encoding→IrOp) | `xps5x-gpu` | `wip` | `0e11786` | verified `lift_to_ir`: the encoding→IrOp table (VOP2 Add/Mul/IAdd, VOP1 Mov/Sqrt, SOP2/SOP1), sequential SSA result numbering, EXP/S_ENDPGM as sinks with no SSA result, and resource counting (SMEM→ubo, MIMG→texture, VINTRP→input, EXP→output). Unknown opcode within a known encoding → Nop (no panic). Remaining: operand (src/dst) wiring into IR `sources`, which then unblocks real SPIR-V body emission |
| emulator/Graphics: VOP1/VOP2 operand decode | `xps5x-gpu` | `wip` | `d0d9c7d` | `decode_operands` fills `Instruction.src/dst` for VOP1/VOP2: the 9-bit SRC0 field (`decode_src9` — SGPR 0-101, VGPR 256-511→0-255, inline int +1..+64/-1..-16, inline float 0.5/±1/±2/±4 as IEEE bits, VCC/EXEC/M0, literal-follows marker), VSRC1, VDST. 4 tests vs known encodings. Unmodeled encodings keep empty operands (honest partial coverage). Remaining: SMEM/MIMG/EXP operand layouts + SSA-correct `sources` wiring (vgpr→last-def map) |
| emulator/Graphics: GCN→IR SSA source wiring | `xps5x-gpu` | `wip` | `9c96214` | `lift_to_ir` threads VOP operands into IR `sources` by local value numbering: a VGPR read resolves to the SSA reg a prior op wrote (`vgpr_def`), an undefined VGPR read is a live-in `Input`, inline int/literal fold to IR constants; each VOP result records its VDST → real def→use chains (verified `v2=v0+v1; v3=v2*v2`). Remaining: a parallel scalar (SGPR) SSA map, SMEM/MIMG/EXP operand decode, then IR→SPIR-V arithmetic body emission (the SPIR-V emitter's `main{ret}` becomes real once sources reach it) |
| emulator/Graphics: scalar (SGPR) SSA path | `xps5x-gpu` | `wip` | `90128c2` | `decode_operands` now decodes SOP2 (SSRC0/SSRC1/SDST) + SOP1 (SSRC0/SDST) 8-bit scalar fields; `lift_to_ir` gains a `sgpr_def` value-numbering map symmetric to the vector one, and `resolve_source` consults both (replacing the SGPR→Input approximation). Verified scalar def→use chain (`s2=s0+s1; s3=mov s2`). Remaining: SMEM/MIMG/EXP operand layouts, then IR→SPIR-V arithmetic body — which is gated on a `spirv-val`/spirv-tools validator in the test harness (structure is testable in-tree; full body *validity* is not, so it won't be emitted-and-claimed without one) |
| emulator/Graphics: EXP export-target decode | `xps5x-gpu` | `wip` | `0eb504a` | `classify_instruction` carries the EXP `TGT[9:4]` as the opcode; `lift_to_ir` maps it to the export kind (MRT0-7/MRTZ→`ExportColor`, POS0-3→`ExportPosition`, PARAM0-31→`ExportParam`) instead of a hardcoded color export. 3 targets verified. Remaining: the four `VSRC` export operands live in word1 → needs the 64-bit `raw` threaded into operand decode |
| emulator/Graphics: EXP VSRC operand decode | `xps5x-gpu` | `wip` | `902ed97` | `decode_operands` now takes the full 64-bit `raw`; the EXP arm reads `EN[3:0]` (word0) and pulls the enabled `VSRC0..3` VGPRs from word1, and `lift_to_ir` threads them through the SSA maps so an exported VGPR references the value that produced it (verified `v2=v0+v1; export v2` → source is the add's SSA reg; partial EN masks select the right subset). Vector/scalar ALU + EXP now all carry real SSA operands. Remaining front-end: SMEM/MIMG operand layouts, then IR→SPIR-V arithmetic body (gated on a `spirv-val` validator in the harness) |
| emulator/Graphics: SMEM operand decode | `xps5x-gpu` | `wip` | `d563a77` | `decode_operands` SMEM arm decodes `SBASE[5:0]` (descriptor SGPR-pair base, ×2) as a source and `SDATA[12:6]` as the destination SGPR; `lift_to_ir` resolves the descriptor through the SSA maps and records the `BufferLoad` result into `sgpr_def` so a later scalar read of that SGPR chains to the load. Verified field extraction. **Front-end now decodes + SSA-threads VOP1/2, SOP1/2, EXP, SMEM.** Remaining: SMEM offset (SOFFSET/imm in word1), MIMG/MUBUF/MTBUF operand layouts, then IR→SPIR-V arithmetic body (gated on a `spirv-val`/spirv-tools validator in the test harness) |
| emulator/Graphics: PM4 command **encoder** (`Pm4Writer`) | `xps5x-gpu` | `done` | `db50b13` | in-house PM4 build side — the exact inverse of `Pm4Header::from_raw`/`process_command_buffer`. `set_context_reg`/`set_sh_reg`/`draw_index_auto`/`dispatch_direct`/`write_type0`/`nop`/`type3`. Round-trip verified: encoder output decodes to the right register values + draw/dispatch/packet counts. This is the GPU-command-emission foundation the `libSceAgc` port (66 DCB/ACB PM4 emitters) builds on — kept as clean round-trip-verified code rather than a low-confidence transliteration of Agc's command-buffer object model. 3 tests. |
| emulator/Graphics: libSceAgc DCB emitters (cursor + PM4) | `xps5x-hle` | `wip` | `67d3399`,`1745228`,`217d4f7`,`3a8d787`,`ee492b2`,`3653788`,`3620a10`,`c496e06`,`c54561e`,`dbbd5e4`,`7545443`,`8a4f09e`,`0c28932` | `libsce_agc.rs` — port of SharpEmu Agc's Draw-Command-Buffer cursor model (`alloc_command_dwords`: CB struct cursor-up@0x10/down@0x18/reserved@0x30) + **15 functions**: `Init` + 19 command emitters + `CreatePrimState`+`CreateShader`+`CreateInterpolantMapping` across DCB/Cb/Acb (draws, index setup, instances, markers, dispatch, flip, resets, nop, wait-flip, SH-register-range copy, event-write), emitting PM4 in Agc's dialect (`0xC000_0000\|(len-2)<<16\|op<<8\|(reg&0x3F)<<2`). Verified vs SharpEmu's exact bytes + cursor advance. 17 tests. **Verifiable boundary:** the 23 ported are every Agc function that's a clean, fixed-layout, register-args-only packet emitter (verifiable vs exact bytes). **Correction:** struct-writers whose offsets the reference specifies (e.g. `CreatePrimState`, `7545443`) ARE portable + testable-vs-reference, and are being done. What genuinely stays blocked are only:  and are the *same category as the infrastructure gates* — (a) the large hardware **register-defaults tables** (`GetRegisterDefaults2*`) — unverifiable data; (c) **variable-length sync packets** (`*AcquireMem`/`*WaitRegMem`/`*ReleaseMem`/`*DmaData`) that read overflow args from the **guest stack**, an ABI my register-args HLE path can't confirm. These + `Ampr` land properly with the Vulkan submit path (real consumer to verify against) — not as guessed transcriptions. |
| emulator/Graphics: GCN→IR body + Vulkan submit→present | `kyty-graphics` → `xps5x-gpu` | `todo` | | crown jewel / M2+ — real shader translation body + Vulkan draw + swapchain present; needs verification vs real render output. Front-end decode→IR→SPIR-V-header is now all test-covered (`9cee071`,`3acad0f`,`0e11786`,`ddedfab`); the gap is operand decode (populates IR `sources`) → SPIR-V arithmetic body → Vulkan. Guest-facing PM4 emission started: `Pm4Writer` (`db50b13`, gnm dialect) + `libSceAgc` DCB emitters (`67d3399`, Agc dialect). |
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
| Sysmodule load chain (system modules, all HLE) | Libs | `xps5x-hle` | `done` | `e2e9eb2` | sceSysmoduleLoadModule/LoadModuleInternalWithArg/Unload/IsLoaded all succeed — every SCE_SYSMODULE_* is HLE-registered, so "loading" resolves without bringing anything into memory. Panic-safe arg access + test |
| Real user .prx load chain (file-backed) | Kernel / Libs | `xps5x-firmware`/`hle` | `todo` | | a title's own split-out .prx from disk — needs a module-load service reachable from HleContext (the M1-D architectural blocker) |
| Np (PSN) manager — offline | Np | `xps5x-hle` | `done` | `df2ad63` | new libsce_np.rs: sceNpGetState reports SIGNED_OUT (SharpEmu value 1), reachability UNAVAILABLE, GetOnlineId="Player", callbacks accepted-but-never-fire — a title sees "offline", disables online features, and boots instead of hanging on a PSN check. Online Np (trophy sync, matchmaking) out of scope. 2 tests |
| Fiber / AMPR | Kernel | `xps5x-hle`/`runtime` | `todo` | | verified blocked on infrastructure, not stubbable: sceFiberRun/Switch transfer control to *run the fiber's guest entry* → needs the guest execution-context machinery (M1-E family) — an init-only stub is the "no-op instead of real execution" anti-pattern. AMPR command buffers ultimately need a DMA/compute processing backend (overlaps M2). |
| Mouse / Ime / GameUpdate / SystemGesture / Share / NpGameIntent (headless) | Libs | `xps5x-hle` | `done` | `3de1121`,`410e985` | libsce_peripheral.rs: libSceMouse (Open/Close/Read → 0 entries), libSceIme (Update/Keyboard* → no pending events), libSceGameUpdate (Init/Terminate OK), libSceSystemGesture (8 NIDs — recognizers OK, GetTouchEventsCount → 0), libSceShare (Init/SetContentParam OK), libSceNpGameIntent (Init OK). Each reports its benign headless "nothing here" state, resolving NIDs real titles poll (Quake KEX calls `sceImeUpdate` each frame) that were unresolved → crash. SharpEmu-cross-checked. 1 test. |
| Media handshakes — Ajm / Ngs2 / AvPlayer / Ult | Libs | `xps5x-hle` | `done` | `143f741` | new libsce_media.rs: faithful port of SharpEmu's Ajm/Ngs2/AvPlayer/Ult (themselves handshake stubs — no real DSP in the reference). Ajm init (context id), Ngs2 system/rack/voice-handle mgmt + idle VoiceGetState, AvPlayer Init(null handle)/IsActive(inactive)/GetVideo/AudioData(no frame)/Close, Ult init. Subsystems initialize so titles proceed; **no real media output yet** (same shape as AudioOut/VideoOut). Real decode/synthesis = follow-up. 3 tests. |
| Json (`sce::Json` C++ lifecycle) + VoiceQoS | Libs | `xps5x-hle` | `done` | `10dfa92` | new libsce_json.rs: port of SharpEmu's `sce::Json` exports — mangled MemAllocator/Initializer/InitParameter2 ctors (return `this`), dtors (return 0), fluent `setAllocator`/`setFileBufferSize` (return `this`), `Initializer::initialize` (OK / EINVAL for null this); registered under `libSceJson`+`libSceJson2`. C++-ABI-correct lifecycle (parsing itself not exercised by these NIDs). Plus `libSceVoiceQoS` `sceVoiceQoSInit` (no voice backend). 3 tests. |
| Rtc — calendar math + clock (complete) | Libs | `xps5x-hle` | `done` | `726ab2c`,`bd9d79e`,`5407aa7` | new libsce_rtc.rs: port of SharpEmu RtcExports. 25 NIDs — GetTickResolution/IsLeapYear/GetDaysInMonth/GetDayOfWeek (Sakamoto, Sun=0), CheckValid (field-by-field `SceRtcDateTime` validation w/ Orbis error codes), TickAdd{Ticks/µs/Sec/Min/Hour/Day/Week} (signed µs tick arithmetic, over/underflow→error), CompareTick(-1/0/1), GetCurrentTick + Network/RawNetwork/AdNetwork/DebugNetwork aliases (host UTC as an Rtc tick), **GetCurrentClock/LocalTime** (tick→`SceRtcDateTime` via Hinnant civil-from-days). Tick = µs since 0001-01-01. 7 tests. **libSceRtc complete.** |
| PlayGo | Libs | `xps5x-hle` | `done` | `39f7e58` | new libsce_playgo.rs: all chunks LOCUS_LOCAL_FAST, progress==total (complete), empty to-do list, handle out — "everything installed" so titles skip download gating (SharpEmu-cross-checked values). 13 NIDs; 3 tests |
| UserService (initial user / login list / name / event) | UserService | `xps5x-hle` | `done` | `3d44b94` | new libsce_user_service.rs: single local user id 1000 (SharpEmu PrimaryUserId), GetInitialUser/GetLoginUserIdList/GetUserName("Player")/GetEvent(NO_EVENT) — supplies the userId scePadOpen/save-data need. 6 NIDs; 3 tests |
| SystemService (status / param / safe-area) | SystemService | `xps5x-hle` | `done` | `46432e4` | new libsce_system_service.rs: GetStatus(eventNum=0, quiet), ParamGetInt (SharpEmu mapping 1/2/3/1000→1, 4→180), GetDisplaySafeAreaInfo(ratio 1.0), HideSplashScreen/ReportAbnormalTermination OK — a title's per-frame poll runs undisturbed. 5 NIDs; 3 tests |
| pthread **mutexes** (state machine) | Kernel | `xps5x-hle` | `done` | `cf265b6` | `pthread_sync.rs` — faithful port of SharpEmu's mutex state machine (type/owner/recursion) into per-process kernel state (`OrbisKernel::pthread_mutexes`/`pthread_mutex_attrs`, keyed by both `pthread_mutex_t` addr and allocated handle). Init writes a real opaque handle; Lock/Trylock/Unlock honor recursive (count) / error-check (`EDEADLK`/`EBUSY` on self-relock) / normal-adaptive (lenient) semantics; Destroy + mutexattr Init/Settype/Destroy. Correct for single-active-execution (1 guest thread → ownership = recursion + type). 5 tests. Adds Trylock/Destroy/mutexattr* (new NIDs). Replaces the return-0 stubs. |
| **BSD sockets** (offline) + net helpers (`socket`/`bind`/`connect`/`getsockname`/`htons`/`inet_pton`/`bzero`) | Kernel | `xps5x-hle` | `done` | `574447f` | `kernel_socket.rs` — port of SharpEmu's `KernelSocketCompatExports` **minus its real host-TCP path** (XPS5X models no host connectivity; guest code must never reach the host network). socket/bind/getsockname track offline state (`OrbisKernel::kernel_sockets`, fds in a high range); connect always fails. Pure helpers fully correct: htons (byte swap), inet_pton (dotted-quad→4 bytes), bzero (zero guest mem). 4 tests. Security note: deliberately no host TCP. |
| **Kernel event queues** + user events (`sceKernelCreateEqueue`/`AddUserEvent`/`TriggerUserEvent`/`WaitEqueue`/`GetEvent*`) | Kernel | `xps5x-hle` | `done` | `e2b3432` | `kernel_equeue.rs` — port of the user-event core of SharpEmu's `KernelEventQueueCompatExports` into `OrbisKernel::kernel_equeues`/`kernel_equeue_events`. Replaces libkernel's fake-handle-1 stub. Create/Delete, AddUserEvent[Edge]/DeleteUserEvent, TriggerUserEvent (pending+udata), WaitEqueue (delivers 32-byte `SceKernelEvent` structs — ident/filter=EVFILT_USER/fflags/udata — edge-clears, writes count; ETIMEDOUT if none), GetEventId/Filter/Data/UserData. Registration/trigger/delivery **fully correct**; true blocking wait + AMPR/graphics events need M1-E/M2. 3 tests. |
| **Kernel semaphores** (`sceKernelCreate/Signal/Wait/Poll/Cancel/DeleteSema`) | Kernel | `xps5x-hle` | `done` | `53c78ee` | `kernel_semaphore.rs` — port of SharpEmu's `KernelSemaphoreCompatExports` into `OrbisKernel::kernel_semaphores` (count + max per handle). Create (validation, count=init, u32 handle out), Signal (+count, error if > max), Poll (dec if available else EBUSY), Wait (dec if available else ETIMEDOUT), Cancel (reset + 0 waiters), Delete. Count arithmetic **fully correct**; true blocking Wait needs the M1-E scheduler. SCE error codes. 5 tests. Previously-unimplemented NIDs. |
| **Kernel event flags** (`sceKernelCreate/Set/Clear/Poll/Wait/Cancel/DeleteEventFlag`) | Kernel | `xps5x-hle` | `done` | `92b1fdf` | `kernel_eventflag.rs` — port of SharpEmu's `KernelEventFlagCompatExports` into `OrbisKernel::kernel_event_flags` (64-bit condition bits per handle). Create (attr/name validation, initial bits, handle out), Set (OR), Clear (`&= pattern`, Orbis semantics), Poll (AND/OR + CLEAR_ALL/PATTERN, EBUSY if unmet), Wait (OK if satisfied else ETIMEDOUT), Cancel (force bits + 0 waiters), Delete. Bit state **fully correct**; true blocking Wait needs the M1-E scheduler. SCE error codes. 5 tests. Previously-unimplemented NIDs. |
| pthread **thread identity/control** (`Self`/`Equal`/`Yield`/`Rename`/`Getthreadid`) | Kernel | `xps5x-hle` | `done` | `a3e1170` | `pthread_thread.rs` — the small stateless pthread calls a title makes constantly (SharpEmu-cross-checked). `Self`/`Getthreadid` return the one guest thread handle, `Equal` compares, `Yield` succeeds (nothing to switch to), `Rename` accepts + logs. **Complete and exact** for single-active-execution. 3 tests. All previously-unimplemented NIDs (were unresolved → crash). |
| pthread **TLS keys** (`scePthreadKey*`/`Set`/`Getspecific`) | Kernel | `xps5x-hle` | `done` | `d1f4e1b` | `pthread_tls.rs` — port of SharpEmu's pthread-key TLS into `OrbisKernel::pthread_tls_keys`/`pthread_tls_values`. KeyCreate allocates a monotonic key (+ destructor) and writes it to `*key`; Setspecific stores a per-thread value (EINVAL for unknown key); Getspecific returns it (0 if unset/unknown, value IS the return per ABI); KeyDelete drops the key + all its values. **Fully complete** — the (thread, key) map is exact under single-active-execution, no runtime dependency. libc/runtimes use this for thread-local storage. 4 tests. All previously-unimplemented NIDs. |
| pthread **thread attributes** (`scePthreadAttr*`) | Kernel | `xps5x-hle` | `done` | `7908c41` | `pthread_attr.rs` — port of SharpEmu's `PthreadAttrState` into `OrbisKernel::pthread_attrs` (detach/stack/guard/sched, keyed by addr + handle). Init writes a handle + default state (1 MiB stack, joinable); Set/Get detachstate/stacksize/guardsize + Set schedpolicy round-trip (Get writes to the guest out-ptr); Destroy clears. **Fully complete** — thread attributes are pure config with no runtime dependency (unlike thread *creation*). 3 tests. All previously-unimplemented NIDs. |
| pthread **rwlocks** (state machine) | Kernel | `xps5x-hle` | `done` | `4883277` | `pthread_sync.rs` — port of SharpEmu's `PthreadRwlockState` into `OrbisKernel::pthread_rwlocks` (readers / writer / writer_recursion, keyed by addr + handle). Init writes a real handle; Rdlock/Tryrdlock nest read holds, Wrlock/Trywrlock acquire-or-recurse the write hold, Unlock releases write-then-read (`EPERM` if neither held); Destroy + rwlockattr Init/Destroy. Correct for single-active-execution. 3 tests. New NIDs (was fully unimplemented). |
| pthread threads / create-join / cond blocking | Kernel | `xps5x-runtime`/`hle` | `todo` | | M1-E — `scePthreadCreate` running the thread entry + `CondWait`/blocking need the per-thread guest scheduler (a real 2nd guest context); an init-only stub is the "no-op instead of real execution" anti-pattern. Mutex + rwlock *state* is now done (above). |
| AudioOut (pacing stub, no playback yet) | Audio | `xps5x-hle` | `done` | `bedb5e0` | libsce_audio_out.rs: Init/Open(handle+grain/freq)/Output(acks buffer, sleeps ~grain÷freq bounded, returns sample count)/Close/SetVolume — audio thread paces without hang or 100% spin (M3 "audio must not hang"). Real host playback (cpal) is the follow-up. 3 tests |
| VideoOut: flip-completion + resolution (no real present yet) | Graphics | `xps5x-hle` | `done` | `f908ddb` | SubmitFlip bumps a flip counter + records flipArg; GetFlipStatus reports count + zero-pending so the render loop advances (was stalling); GetResolutionStatus=1080p, GetVblankStatus=frame counter. Real swapchain present = M2/M3 follow-up. 2 tests |
| VideoOut real present / AGC / shaders→Vulkan | Graphics | `xps5x-gpu`/`hle` | `todo` | | M2+ — the actual GPU pipeline (crown jewel) |
| DualSense / pad (digital+analog state) | Input | `xps5x-input`/`hle` | `done` | `0ceb7db` | ControllerState→Orbis ScePadData encoder (documented button masks, stick/trigger byte mapping) in xps5x-input; scePadReadState writes a valid state + returns 1 (was garbage + 0 → homebrew read-loop hang). Live host-input routing (InputManager→HleContext) + haptics/adaptive-triggers still todo |
| DualSense: live input routing (kernel snapshot → scePadReadState) | Input | `xps5x-kernel`/`hle` | `done` | `cb1b56d` | kernel holds a settable 12-byte pad-state snapshot (OrbisKernel::set_pad_state/pad_state); scePadReadState reads it (neutral fallback when unset) — live host input now flows guest-ward. Remaining: Shell polling InputManager into set_pad_state each frame (UI wiring) + haptics/adaptive triggers |
| Filesystem: open/read/close/lseek | KernelExports/FS | `xps5x-kernel`/`hle` | `done` | `896495d` | VFS-backed, real host files under /app0; write persistence + fstat still todo |
| Filesystem: write persistence (savedata) | FS | `xps5x-kernel`/`hle` | `done` | `5285857` | VFS honors O_WRONLY/RDWR/CREAT/TRUNC/APPEND; write buffers + flush-on-close to host file; ".." traversal refused on writable open; hle write() routes non-console fds to VFS; hle open() honors O_CREAT. E2E: guest open+write+close persists to host, read-back works |
| Filesystem: fstat / directory ops | FS | `xps5x-kernel`/`hle` | `todo` | | needs SCE stat struct layout |
| SaveData mount | SaveData | `xps5x-hle` | `done` | `93a7c0a` | new libsce_save_data.rs (was empty stub): sceSaveDataMount{,2,3} writes the /savedata0 mount point into the 64-byte result; Umount/Initialize/Terminate OK. Completes the save path — mount → open/write under /savedata0 → VFS persists to host savedata dir (write-persistence 5285857). SharpEmu-cross-checked. 2 tests |
| CommonDialog / MsgDialog | CommonDialog | `xps5x-hle` | `done` | `264d895` | new libsce_common_dialog.rs: no host popup → sceMsgDialogOpen completes immediately (status→FINISHED), GetResult reports the OK button, so a title's dialog poll loop finishes instead of hanging. Status None/Init/Finished per SharpEmu. 2 tests |
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
