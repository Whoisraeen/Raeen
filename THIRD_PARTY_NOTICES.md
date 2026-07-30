# Third-Party Notices

Raeen is licensed under **GPL-2.0-only** (see [`LICENSE`](LICENSE)). It
incorporates ideas and re-implemented code derived from the third-party
projects listed below. Raeen ships **no Sony code, keys, or firmware**; every
subsystem here is original Rust, written clean-room with respect to Sony's
proprietary SDK/headers, using only the community sources credited below.

---

## Kyty — PS4/PS5 emulator (reference & porting source)

- **Upstream:** https://github.com/InoriRus/Kyty
- **License:** MIT
- **Copyright:** © 2021 InoriRus
- **How Raeen uses it:** Portions of Raeen's GPU (GNM command-buffer parsing,
  PSSL → SPIR-V shader translation, RDNA/GCN → Vulkan) and HLE kernel/library
  layers are **re-implemented in idiomatic Rust with reference to Kyty's C++
  source.** Kyty's tree is cloned locally into the git-ignored `reference/`
  directory for study only; it is **never vendored, compiled, or committed**
  into Raeen. The MIT license permits use, modification, and redistribution of
  such derived work provided the copyright notice below is retained — which
  this file does.
- **Directly ported data:** the AGC Gen5 register-default tables in
  `crates/raeen-hle/src/libsce_agc_reg_defaults.rs` (served by
  `sceAgcGetRegisterDefaults2[Internal]`) are a faithful port of Kyty's
  `Graphics.cpp` `g_cx/sh/uc_reg_info1/2` tables with register names resolved
  against Kyty's `Pm4.h`.
- **Behavioral mapping:** `raeen-gpu`'s Gen5 stencil conversion follows Kyty's
  explicit AMD-operation mapping rather than treating AMD and Vulkan enum
  values as layout-compatible. Raeen's Rust implementation additionally
  validates unsupported operations and preserves the guest's separate test and
  operation reference values.
- **Divergence recorded (vertex-input numeric classes, 2026-07-28).** Kyty's
  `Spirv::WriteGlobalVariables` (`ShaderSpirv.cpp` L7229) declares vertex
  attributes as **float only** and `EXIT`s on any other width; it has no
  integer vertex-input concept. `kyty-graphics`' shared
  `vertex_input_types` resolver therefore goes beyond Kyty for the raw
  integer classes (see the SharpEmu section below), while keeping Kyty's
  structure and `fetch_*` helper functions unchanged.

MIT is compatible with GPL-2.0: MIT-derived portions may be combined into this
GPL-2.0-only work, and this notice preserves the required MIT attribution for
those portions.

```
MIT License

Copyright (c) 2021 InoriRus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## KytyPS5 — active PS5 Kyty fork (reference & porting source)

- **Upstream:** https://github.com/Nmzik/KytyPS5
- **License:** GPL-2.0 with Kyty/MIT lineage
- **How Raeen uses it:** PS5-specific GPU, VM, pthread, and HLE behavior is
  compared selectively rather than merged wholesale. Raeen's runtime-owned
  pthread allocation adds a separate 1 MiB reserve after observing Minecraft
  enter a fixed 0x14a778-byte stack frame on a thread whose guest-visible
  attribute remains 1 MiB. This is an idiomatic Rust re-implementation of
  KytyPS5's separate `PTHREAD_STACK_EXTRA` behavior. Raeen also follows
  KytyPS5's Gen5 vertex-attribute handling by carrying the AGC `fetch_index`
  selector into Vulkan's per-vertex/per-instance input rate; the Rust data
  model, cache keys, and tests are original. AudioOut2's context-memory sizing
  and speaker-array lifecycle/coefficient behavior are likewise behaviorally
  re-implemented in Rust from `src/libs/libAudio2.cpp`; this closes GTA V's
  measured undersized-allocation and unresolved speaker-array initialization
  path. GTA V's Gen5 AGC fixed packet sizes, direct Cx/Sh/Uc register writers,
  and DCB/ACB conditional-execution packet layout are behaviorally
  re-implemented from `src/libs/agc.cpp`; SharpEmu independently confirms the
  by-value `{u32 offset, u32 value}` direct-register ABI. The GTA V Phase A
  AGC batch (`crates/raeen-hle/src/libsce_agc.rs`, 2026-07-27) extends this to
  further `src/libs/agc.cpp` behaviors: the 14-DWORD `CbBranch` conditional
  chain, `DcbRewind`, the 18/12-DWORD workload active/complete markers,
  `DcbDrawIndexMultiInstanced`, `UpdatePrimState` (with
  `GraphicsPrimitiveTypeToGsOut`), `GetDataPacketPayloadRange`, the
  CondExec/QueueEndOfPipeAction/WaitRegMem/DmaData packet-patch family, and
  the `*GetSize` byte counts pinned to Kyty's Gen5 emitters. The Phase B ACB
  execution batch (2026-07-27) behaviorally re-implements three further
  `src/libs/agc.cpp` mechanisms in `crates/raeen-hle/src/libsce_agc.rs`: the
  5-DWORD ACB submission-descriptor indirection (`submit_acb`, magic
  `0x5533ccaa`), the pending post-submit graphics-segment tracker
  (`track_pending_graphics_segment_after_submit` /
  `track_pending_graphics_allocation` in `CommandBuffer::AllocateDW`), and
  `flush_pending_graphics_segment_before_acb` (ACB wait-address collection,
  RELEASE_MEM producer matching, packet-boundary trimming, flush-as-DCB).
  The command-buffer grow path (2026-07-29) follows KytyPS5's
  `CommandBuffer::ReserveDW` callback ABI and one-shot post-callback capacity
  check, independently cross-checked against SharpEmu's
  `TryAllocateCommandDwords`.
  `crates/kyty-graphics/src/run.rs` additionally re-implements
  `CpOpDispatchIndirect` (both the base+offset and absolute-address argument
  forms, `pm4Handlers.cpp`/`graphicsRun.cpp`) and `CpOpSetBase`'s shader-type
  split between the indirect-draw and indirect-dispatch argument bases. The
  PM4 decoder-agreement batch (2026-07-29) takes the `IT_DISPATCH_DRAW`
  opcode NUMBER (`0x8D`) from `src/graphics/guest_gpu/pm4.h` L71, plus the
  factual observation that `MakeOpcodeDispatchTable` (`pm4Dispatch.cpp` L212)
  wires only `IT_DISPATCH_DRAW_PREAMBLE` (`0x3A`) and never `0x8D` — that
  absence is why Raeen refuses the opcode by name instead of guessing a body
  layout. No handler logic was ported for `0x8D`.

  The follow-up batch the same day DOES port `0x3A`:
  `CommandProcessor::cp_op_draw_index_multi_instanced`
  (`crates/kyty-graphics/src/run.rs`) re-implements the `0xC0073A00` branch of
  `CpOpDrawIndex` (`pm4Handlers.cpp` L2276-2297) in Rust — the field order
  `[index_count, addr_lo, addr_hi, max_instance_count, obj_lo, obj_hi,
  instance_count, flags]` and the 8-body-dword length, matching the opcode
  number from `pm4.h` L44. Raeen deviates in three ways: KytyPS5 `EXIT`s on any
  other `cmd_id` where Raeen issues a counted, named refusal (resilience
  policy); Raeen's `IndexedDraw` carries no instance count or object-id buffer,
  so the multi-instanced part degrades to one instance with a rate-limited warn;
  and all tests are Raeen's own. The matching emitter,
  `raeen-hle::hle_dcb_draw_index_multi_instanced`, was already attributed to
  `GraphicsDcbDrawIndexMultiInstanced` above. No C++ is vendored or compiled
  into Raeen.
  the `*GetSize` byte counts pinned to Kyty's Gen5 emitters. The GTA V
  `libSceAmpr` Tier-C batch (`crates/raeen-hle/src/libsce_ampr.rs`,
  2026-07-27) behaviorally re-implements `src/libs/libAmpr.cpp`: the
  nop/marker/wait command family as inert records, the KytyPS5 argument
  bounds (`APR_MAX_*`, 16 KiB map granularity, 1..=16 nop dwords), the
  gather/scatter read-stream continuation (file id sticks, destination and
  file offset continue past each read), the `WriteAddressFrom*` value-0
  completion, and the APR map-window `EPERM` state machine; record byte
  layouts, the eager-read model, and all tests are Raeen's own. The local
  trophy-store batch (2026-07-28) cross-checks `src/libs/libNet.cpp`:
  `LibNpTrophy2`'s prototypes and struct sizes (`NpTrophy2GameDetails` 152 B,
  `NpTrophy2Details` 1312 B static-asserts), the confirmed
  `NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND` (`0x80553911`) icon-getter shape
  (`*size = 0`), and `LibNpUniversalDataSystem`'s field-proven UDS prototypes
  (`CreateEvent(eventName*, prop, newEvent**, propPtr**)`, property setters
  `(object, key, value)`) re-implemented in
  `crates/raeen-hle/src/libsce_np_trophy2.rs` and
  `libsce_np_universal_data.rs`; KytyPS5's fabricated one-bronze-trophy
  "Kyty" game info is deliberately not ported. The instruction-coverage batch
  (`crates/kyty-graphics/src/shader/`, 2026-07-28) uses
  `src/graphics/shader/recompiler/MemoryOps.cpp`'s `SMEM_OPS` table and
  `ImageOps.cpp`'s `MIMG_GATHER_OPS` table purely as **opcode-identity
  evidence** — that SMEM/SMRD opcode `0x04` is `s_load_dwordx16` (16 dwords),
  that MIMG `0x47`/`0x61` are `image_gather4_lz`/`image_gather4h`, and that a
  gather's destination is always 4 dwords with a single-bit dmask
  (`data_dwords = gather ? 4 : CountDmaskComponents(dmask)` plus
  `IsSingleDmaskBit`). The Rust decode arms, the SPIR-V `OpImageGather`
  component mapping, and all tests are Raeen's own. The SMEM
  register-soffset batch (`crates/kyty-graphics/src/shader/`, 2026-07-28)
  additionally uses `src/graphics/shader/recompiler/MemoryOps.cpp`'s
  `DecodeSmem` (the immediate offset and the soffset operand are stored as two
  INDEPENDENT simultaneous fields — `inst.offset = SignExtendU32(word1 &
  0x1fffff, 21)` alongside `DecodeScalarSource(soffset, ..., inst.src1)`) and
  `spirvEmitter/spirvEmitterMemory.cpp`'s `EmitRelativeAddress` /
  `EmitSLoadDword` as **addressing-rule evidence**: the scalar-load address is
  `base + soffset + immediate` in bytes, dword-aligned by masking the low two
  bits (`align_components`), with the dword index taken as `address >> 2`.
  Raeen's three-operand instruction format, the analysis-side soffset
  resolution, the honesty flag on the raw EUD window, and all tests are its
  own. The **`sceKernelDlsym` semantics** batch
  (`crates/raeen-hle/src/libkernel.rs`, 2026-07-28) uses
  `src/loader/runtimeLinker.cpp`'s `RuntimeLinker::FindProgramById` (L1532 —
  "Id 0 is reserved for main program", `unique_id` handed out from 1) and
  `src/libs/libKernel.cpp`'s `KernelDlsym` (L226) as **contract evidence**:
  handle 0 names the executable rather than a POSIX `RTLD_DEFAULT` global
  scope, a miss is `ESRCH` (not `ENOENT`), and `scriptingGetMem` is answered by
  an emulator-supplied aligned allocator rather than any guest export — its
  `(alignment, size)` signature, the clamp of alignment up to `0x10`, and the
  power-of-two rejection come from `KernelApplicationHeapGetMem` (L203). The
  same file's `AddLibkernelUnityFunc` (L2998, L3081-3088) is the evidence that
  the entire `libkernel_unity` surface is three functions
  (`Qhv5ARAoOEc`/`WkwEd3N7w0Y`/`il03nluKfMk` = Remove/Install/RaiseException),
  all three already registered by Raeen. Raeen's load-ordered module sweep, the
  trampoline-reservation mechanism, the miss diagnostics, and all tests are its
  own. The **`.native` library-name canonicalization** batch
  (`crates/raeen-firmware/src/registry.rs`, 2026-07-28) uses `src/libs/libs.h`'s
  `LIB_VERSION(library, lv, module, mv1, mv2)` macro (L24) and its `.native`
  registrations as **naming-rule evidence**: `libDialog.cpp:87`
  (`"SaveDataDialog.native"` → module `"SaveDataDialog"`), `libDialog.cpp:108`
  (`"MsgDialog.native"` → module `"MsgDialog"`), and `dialog.cpp:497`
  (`LIB_NAME("MsgDialog.native", "MsgDialog")`) together establish that
  `.native` is a spelling of one library whose module identity is the bare
  name, and that both spellings dispatch to the same implementation
  (`Dialog::SaveDataDialog::*`). The **guest working directory** used when
  anchoring relative guest paths (`crates/raeen-kernel/src/filesystem/mod.rs`,
  2026-07-28) is likewise taken from `src/main.cpp:138` /
  `src/emulator.cpp:201-202` (executable loaded as `/app0/eboot.bin`, app
  directory mounted at `/app0` and `/hostapp`);
  `src/kernel/fileSystem.cpp:226-245`'s `MountPoints::GetRealFilename`
  fall-through — returning an unmatched guest path verbatim as a host path — is
  documented as deliberately **not** adopted. `src/libs/audio.cpp:564-582`'s
  `Audio::AudioInInput` simulated grain delay is behavioral evidence for the
  capture-port pacing in `crates/raeen-hle/src/libsce_audio_in.rs`, and
  `libDialog.cpp:114-116` for the `sceMsgDialogProgressBar*` trio. Raeen's
  canonicalization function, the sandboxed path normalization, the port table,
  and all tests are its own. The **free-running host vblank source**
  (`crates/raeen-hle/src/host_vblank.rs`, 2026-07-28) behaviorally
  re-implements the structure of KytyPS5's guest-independent display tick:
  `src/graphics/presentation/window/window.cpp:350-354` (`GameShowWindow`
  calling `VideoOutBeginVblank()` → `VideoOutFlipWindow(0)` →
  `VideoOutEndVblank()` once per displayed host frame),
  `src/graphics/presentation/videoOut.cpp:649-686`
  (`VideoOutContext::VblankBegin`/`VblankEnd` advancing the pre-vblank and
  vblank counters and calling `TriggerVideoOutEventsLocked` for **every opened
  handle**), and `videoOut.cpp:402` (`WaitForNextVblank` pacing against
  `Config::GetVblankFrequency()`). Raeen's Rust module, the `&dyn
  WaitSubsystem` equeue-wake seam that makes the tick reachable without an
  `HleContext` (`kernel_equeue::wake_equeue_via`,
  `libsce_video_out::trigger_vblank_events_via`), the single-owner rule that
  stands the guest-driven advances down, the `Weak<OrbisKernel>` teardown, the
  `RAEEN_HOST_VBLANK` gate, and all tests are its own; see
  `docs/host-vblank-source.md`. No C++ is vendored or compiled into Raeen.

  **Signal delivery into a blocking wait** (`crates/raeen-hle/src/exception.rs`
  `deliver_at_wait_slice`/`wake_target_for_exception` and the wait sites that call
  them, 2026-07-29) re-implements the *structure* of KytyPS5's pending-signal
  polling in Rust:
  `src/libs/libKernel.cpp::KernelDispatchPendingSignalForCurrentThread` as a
  per-thread chokepoint the *target* calls, invoked from inside each wait's slice
  loop (`src/kernel/semaphore.cpp:238` and `:430`, `src/kernel/pthread.cpp`'s
  `SleepMicroWithSignalPoll`/`SleepNanoWithSignalPoll`) with the wait's own lock
  **released** across the dispatch (`m_mutex.Unlock()` → dispatch →
  `m_mutex.Lock()`), after which the loop simply continues — i.e. the wait resumes
  rather than returning. `QueuePendingSignal` before the wake, and
  `PthreadWakeForSignal` + `Common::CondVar::SignalThread` as the prompt wake of a
  parked target, are the evidence for Raeen's queue-then-wake ordering and for
  waking a condition waiter without marking it signalled. KytyPS5's Windows
  mechanism (`NtQueueApcThreadEx` special user APC, `SuspendThread` +
  `GetThreadContext` fallback, and dispatching a non-guest context on a helper
  `std::thread`) is **not** adopted: Raeen has no APC path and runs the handler on
  the target's own guest stack via `call_guest`. No C++ is copied.

---

## SharpEmu — PS5 emulator (reference & porting source)

- **Upstream:** https://github.com/sharpemu/sharpemu (formerly par274/sharpemu)
- **License:** GPL-2.0-or-later (compatible with Raeen's GPL-2.0-only)
- **Copyright:** © SharpEmu authors
- **Reference synced:** 2026-07-26 to upstream `main` @ `0535783` (the
  `v0.0.2-beta.5` work is included) — brings working AudioOut2 audio (GTA V),
  AGC cross-queue
  `WAIT_REG_MEM` label work (`Agc/GpuWaitRegistry.cs`), and PR #587's Gen5 flat
  (global) memory + 3D-texture shader support (`Shader/*`, PSSL SEG-field
  FLAT-address decode → SPIR-V global-memory access). Ports cite the
  originating SharpEmu `file:line` in doc comments.
- **`DRAW_INDEX_2` packet layout 2026-07-29**: `hle_dcb_draw_index` in
  `crates/raeen-hle/src/libsce_agc.rs` emitted a five-DWORD `IT_DRAW_INDEX_2`
  whose body zeroed the 64-bit index-buffer base, so the PM4 walker refused
  every indexed draw ("indexed draw with no index buffer: addr=0x0"). SharpEmu
  hit the identical defect on Unity titles and its `DcbDrawIndex`
  (`src/SharpEmu.Libs/Agc/AgcExports.cs` L1567-1610) both documents the cause and
  carries the corrected six-DWORD AMD body
  `{MAX_SIZE, INDEX_BASE_LO, INDEX_BASE_HI, INDEX_COUNT, DRAW_INITIATOR}`.
  Raeen's emitter now matches that layout; the diagnosis is credited to that
  in-source comment.
- **Gen5 shader batch 2026-07-28** (`docs/sharpemu-port/gen5-shader-agc.md`):
  the **VOP3P** (packed 16-bit math) decode and lowering is a behavioral
  re-implementation of `Gen5ShaderTranslator.cs` (`DecodeVop3p`, the
  `Gen5ShaderEncoding.Vop3p` operand/control case), `Gen5ShaderIr.cs`
  (`Gen5Vop3pControl`) and `Gen5SpirvTranslator.Alu.cs` (`TryEmitPackedF16`,
  `TryEmitFmaMix`, `EmitPackedF16Operand`, `EmitPackedF16MinMax`,
  `EmitClampToUnitInterval`, `EmitFmaMixOperand`) — SharpEmu PRs #466
  (`3574a3b`), #460 (`472fc96`), #420 (`3005bab`). Landed as
  `shader_parse_vop3p` in `crates/kyty-graphics/src/shader/parse.rs`,
  `Vop3pControl` in `shader/types.rs`, and
  `recompile_vop3p_packed_f16` / `recompile_vop3p_fma_mix` in
  `shader/recompile.rs`. Two deviations are documented in-tree: the f16↔f32
  conversions use GLSL `Un/PackHalf2x16` rather than SharpEmu's explicit
  integer sequences, and `v_pk_fma_f16` omits the round-to-odd 2Sum (#420)
  because this generator has no per-body `NoContraction` decoration hook.
  The type-driven 3D-image `volume` flag in `crates/raeen-gpu/src/vulkan/`
  and `draw_translate.rs` completes the host half of PR #587 (`5228335`).
- **SMEM register-soffset batch 2026-07-28**
  (`docs/sharpemu-port/smem-register-soffset.md`): the RDNA2 scalar-load
  addressing rule Raeen implements is corroborated against
  `Shader/Gen5ShaderTranslator.cs` (the `Gen5ShaderEncoding.Smem` case keeps the
  sign-extended immediate and an optional `dynamicOffsetRegister` as two
  simultaneous fields of `Gen5ScalarMemoryControl`) and
  `Shader/Gen5ShaderScalarEvaluator.cs` L1889-1900, whose evaluation is exactly
  `byteOffset = immediateOffset + dynamicOffset;`
  `address = (baseAddress + byteOffset) & ~3UL;` — i.e. both offsets are byte
  quantities, they ADD, and the sum is dword-aligned by truncation. That batch's
  soffset resolution was bounded and its own (an unwritten live-in user-data
  register, or a preceding `s_mov_b32`/`s_movk_i32` constant); the general
  evaluator landed in the following batch. Anything unresolved keeps a named
  refusal carrying the width and format. No C# is copied.
- **Gen5 scalar-evaluator batch 2026-07-28**
  (`docs/sharpemu-port/scalar-evaluator.md`):
  `crates/kyty-graphics/src/shader/scalar_eval.rs` is a behavioral
  re-implementation of `Shader/Gen5ShaderScalarEvaluator.cs` — the per-opcode
  semantics of `TryExecuteScalarAlu` (L1118-1575: the SOP1/SOP2/SOPK arms, the
  64-bit `WriteScalarPair` pair model, `SignedAddOverflow`/`SignedSubOverflow`,
  `ReverseBits`, the `s_bfe_*`/`s_bfm_*` offset/width clamping, and which arms
  publish SCC), `TryExecuteSaveExecScalarAlu` (L1577-1675),
  `TryExecuteScalarCompare` (L1742-1803), and the operand model of
  `TryEvaluateScalarOperand`/`TryEvaluateScalarOperand64`/
  `TryDecodeInlineConstant` (L2249-2345 — including the 64-bit sign extension of
  the negative inline-constant range and the VCCZ/EXECZ/SCC special sources).
  Two deliberate deviations, both documented in the module: (1) SharpEmu runs a
  **concrete** interpreter over a seeded 256-entry `uint[]` and errors out on
  anything it cannot execute, whereas Raeen evaluates a two-point
  **known/unknown lattice** so an unmodelled definition or a guest-memory read
  degrades to `Unknown` rather than to a concrete guess — and consequently
  Raeen does not seed `exec` with a full wave mask; (2) SharpEmu forks
  supplemental paths at forward `s_branch` for resource discovery
  (`QueuePath`/`ScalarPathKey`, L142-280), whereas Raeen follows only branches
  whose condition it can prove and refuses undecidable branches, loops and
  indirect PC writes by name. No C# is copied.
  quantities, they ADD, and the sum is dword-aligned by truncation. Raeen's
  soffset resolution is bounded and its own (an unwritten live-in user-data
  register, or a preceding `s_mov_b32`/`s_movk_i32` constant); SharpEmu's
  general abstract scalar interpreter is not ported. Anything unresolved keeps a
  named refusal carrying the width and format. No C# is copied.
- **V#-based buffer loads 2026-07-28**
  (`docs/sharpemu-port/vsharp-buffer-loads.md`): the addressing rule for
  `s_buffer_load*` — whose base is a 4-dword buffer resource descriptor (V#)
  rather than a pointer pair — is established from
  `Shader/Gen5ShaderScalarEvaluator.cs::TryExecuteScalarLoad` L1875-1902, which
  runs the pointer and descriptor families through ONE body where only
  `baseAddress` differs (`hasBufferDescriptor ? bufferDescriptor.BaseAddress :
  <sgpr pair>`) while `byteOffset = immediateOffset + dynamicOffset` and
  `address = (baseAddress + byteOffset) & ~3UL` are shared — i.e. the V#
  contributes only its base address, and the combined soffset+immediate rule
  transfers unchanged. `TryDecodeBufferDescriptor` L2163-2216 supplies the field
  layout (`base48`, `stride = (word1 >> 16) & 0x3FFF`,
  `sizeBytes = stride == 0 ? word2 : stride * word2`, the `word3 >> 30` buffer
  type check, and the `(word3 >> 12) & 0x7F` unified format), which Raeen uses as
  a BOUND and a format selector, never as an address term. Corroborated by
  KytyPS5 (MIT) `MemoryOps.cpp::DecodeSmem` L218-256 and
  `spirvEmitter/spirvEmitterMemory.cpp::EmitBufferByteAddress` /
  `EmitBufferAddressFromParts` L212-278, whose `index * stride` term is gated on
  `idxen` — a flag SMEM does not carry. Raeen's descriptor capture
  (`shader_capture_vsharp_buffer_loads`) is its own bounded pass over live-in
  user SGPRs; SharpEmu's general abstract scalar interpreter and its
  `Gen5GlobalMemoryBinding` machinery are not ported. Unresolved forms keep a
  named refusal carrying type, format, width, descriptor register, soffset and
  immediate. No C# is copied.
- **Exception-delivery model 2026-07-28**
  (`docs/sharpemu-port/loading-blockers.md`): the queue-at-raise /
  deliver-at-the-target's-next-HLE-boundary model in
  `crates/raeen-hle/src/exception.rs` follows the design conclusion in
  SharpEmu's `Cpu/Native/DirectExecutionBackend.cs`
  (`TryRaiseGuestException`, `DeliverPendingGuestExceptionAtSafePoint`,
  `TryWriteGuestExceptionContext`) and `Libs/Kernel/KernelExceptionCompatExports.cs`
  — a raise must not be run on the raising thread, and the target's HLE boundary
  is where that thread is safely paused. The single-slot newest-wins pending
  entry, the per-thread "already delivering" guard, and the 0x500-byte
  `ucontext_t` written at a per-thread scratch address are the same shape.
  Raeen's implementation is original Rust over `OrbisKernel` state and
  `raeen-runtime`'s `call_guest`; no C# is copied.
- **How Raeen uses it:** A second-opinion reference alongside Kyty for PS5
  module loading (eboot/PRX/sysmodule chains), kernel-surface structure
  (Fiber, AMPR, PlayGo), VideoOut/AGC bring-up, and **native host controller
  input** (XInput + raw-HID DualSense — `Host/Windows/WindowsXInputReader.cs`,
  `WindowsDualSenseReader.cs`, `WindowsHidNative.cs`, `Host/HostGamepadState.cs`,
  re-implemented as `crates/raeen-input/src/{xinput,hid,native}.rs`; the
  DualSense **rumble output** path — `BuildOutputReportLocked`'s USB `0x02` /
  Bluetooth `0x31`+CRC-32 framing and the second write-side device handle —
  is likewise re-implemented in `hid.rs` `build_output_report`/`write_loop`,
  rumble-only subset, lightbar/player-LED bytes left untouched). The
  APR/AMPR async file-I/O path is also a SharpEmu port:
  `Ampr/AmprExports.cs` (`AprCommandBufferReadFile` eager read,
  `TryReadFileToGuestMemory` positional read-exact loop + host-handle cache,
  `CompleteCommandBuffer` record walk) and
  `Kernel/KernelAprCompatExports.cs` (submit/wait), re-implemented in
  `crates/raeen-hle/src/libsce_ampr.rs` and the
  `apr_complete_command_buffer` in `crates/raeen-hle/src/libkernel.rs`.
  The Gen5 AudioOut2 context/port parameter layout and grain pacing in
  `crates/raeen-hle/src/libsce_audio_out2.rs`, plus GFX10 MIMG DIM/NSA field
  meanings used by `crates/kyty-graphics/src/shader/parse.rs`, were also
  behaviorally re-implemented after comparison with SharpEmu.
  Type-11 guest cubemaps are lowered to six-layer Vulkan 2D-array views after
  comparing SharpEmu/KytyPS5's `(s,t,face)` image path and Raeen's measured
  Minecraft shader; the Rust implementation and regression tests are original.
  UserService's
  retail-style primary-user id and one-shot login-event behavior were
  re-implemented in `crates/raeen-hle/src/libsce_user_service.rs`; the event
  ABI was independently cross-checked against KytyPS5 and shadPS4. Raeen's
  resource-class-local descriptor indexing, layered guest tiling/writeback,
  and Vulkan staging-size guard are original Rust fixes derived from its own
  Minecraft traces and Vulkan validation output.
  Raeen's executable leaf-import gateway was also designed after auditing
  SharpEmu's native import trampoline state/ABI preservation; Raeen's
  implementation is original Rust plus generated x86-64 code and retains the
  existing VEH route for context-changing calls.
  Raeen's pthread-mutex acquisition queue and direct ownership handoff were
  behaviorally re-implemented from SharpEmu's
  `KernelPthreadCompatExports.cs` waiter/grant model: each waiter has a private
  host wake object, unlock transfers ownership to the FIFO head under the
  mutex-state lock, and arrivals cannot barge ahead. The Windows host-thread
  priority bands in `crates/raeen-runtime/src/thread.rs` follow SharpEmu's
  `DirectExecutionBackend.cs::MapGuestThreadPriority`; KytyPS5
  `src/kernel/pthread.cpp` independently confirms the same Orbis thresholds.
  Both implementations are original Rust and no C# or C++ is vendored.
  From upstream `5228335` (PR #587, "Gen5 flat memory and 3D images"):
  `OpImageQuerySizeLod`'s result vector is sized from the descriptor's
  dimensionality (`%v3int` for 3D and 2D-array, `%v2int` for 2D) in
  `crates/kyty-graphics/src/shader/{spirv,recompile}.rs`, after comparing
  SharpEmu's `Gen5SpirvTranslator.cs`; the Rust emitter and its regression
  tests are original. From upstream `a158960` (PR #592, "GPU compute detile"):
  the row-parallel split of the CPU detile loop in
  `crates/raeen-gpu/src/texture/tiling.rs` follows SharpEmu's `Parallel.For`
  over the same loop in `Agc/GnmTiling.cs`, and the non-power-of-two
  element-size guard mirrors its `BitLog2` refusal. Raeen's swizzle-equation
  tables were verified independently equivalent to SharpEmu's for the modes
  both implement (5/9/24/27) and were **not** changed by that comparison.
  From upstream `6ee445f` (PR #470, "[AGC] Read mip 0 from its GFX10 mip-chain
  offset"): the GFX10 smallest-first mip-chain arithmetic that locates mip 0 at
  the END of a mipped allocation — `GnmTiling.TryGetBaseMipPlacement`'s mip-tail
  capacity/extent thresholds, per-level block-aligned sizes, tail-slot
  Morton-scattered `mipOffset` de-interleave, and
  `TryGetBlockElementDimensions`, plus `TextureDescriptor.ResourceMipLevels` /
  `GetMaximumMipLevels`'s `MAX_MIP + 1` clamp — are re-implemented as
  `base_mip_placement` / `detile_mip_tail_base` / `block_element_dimensions` in
  `crates/raeen-gpu/src/texture/tiling.rs` and `resource_mip_levels` in
  `crates/raeen-gpu/src/draw_translate.rs`. The Rust structure, the diagnostics
  counters, and every test are original; see
  `docs/sharpemu-port/texture-mip-present.md`.
  From upstream PR #422 (`09bd4f0`, "sceKernelSyncOnAddress wait/wake") and
  PR #439 (`73e8821`, "Hand off mutex ownership directly to the head waiter on
  unlock"), both verified present at the live tip: the **guest synchronization
  primitives** in `crates/raeen-kernel/src/lib.rs`
  (`GuestWaiter`/`GuestWaitQueue`/`SyncAddressTable`, `PthreadMutex`'s handoff
  FIFO), `crates/raeen-hle/src/libkernel.rs` (`sync_on_address_wait`) and
  `crates/raeen-hle/src/pthread_sync.rs` (`lock_core`/`hle_mutex_unlock`).
  Re-implemented after studying `Kernel/KernelSyncOnAddressCompatExports.cs`
  (address-keyed park/wake, the wake-count argument reading, the bounded
  self-heal deadline) and `Kernel/KernelPthreadCompatExports.cs`
  (`PthreadMutexUnlockCore`'s handoff-under-lock, `TryGrantMutexWaiterLocked`'s
  FIFO-head grant, `EnqueueMutexWaiterLocked`'s one-entry-per-thread pruning,
  and the fast-acquire refusal of a free-but-queued mutex). Raeen's version
  diverges deliberately in two places: it performs the **futex value compare**
  SharpEmu records as unrecovered (enqueue-then-read ordering, `EAGAIN` on
  mismatch) for the `*Wait32`/`*Wait64` widths, and it replaces the per-address
  wake **generation counter** with a per-waiter FIFO wake bit. The Rust code and
  all its deterministic queue-level tests are original.
  **Out-buffer / stack-canary discipline (2026-07-28).** SharpEmu's out-buffer
  fix series supplied the *rule set* — write exactly the ABI struct size and
  field width, never derive a write length from a guest register, bulk-initialize
  only non-stack objects, do not write reserved/secondary out slots, do not round
  up a size a guest may use as an `alloca` length — after auditing its
  `SharpEmu.Libs/Audio/AudioOut2Exports.cs` (whose own comment records that
  clearing 0x80 bytes "overwrote the caller's stack canary immediately following
  the 0x40-byte parameter block") and `VideoOut/VideoOutExports.cs`. Raeen's
  implementation is original and **structurally different**: SharpEmu detects a
  stack out-pointer with an address-range heuristic over its own import-stack
  window, while `crates/raeen-hle/src/out_buffer.rs` answers the question exactly
  from per-guest-thread stack bounds the runtime registers in
  `raeen_kernel::OrbisKernel::guest_thread_stacks`, falling back to a bounded
  window above `HleContext::caller_rsp`. Two reference sizes were verified
  against SharpEmu's live tree rather than a single commit (`VideoOut` output
  options 0x40 and the `GetOutputStatus` 0x30 layout, which Raeen already
  matched); the audit, the guard API, the Rust code, and all of its tests are
  original. Findings recorded in `docs/sharpemu-port/outbuffer-audit.md`.
  **Gen5 vertex-input numeric classes (2026-07-28).** The model in
  `Gen5SpirvTranslator.DeclareVertexInputs` /`TryEmitVertexInputFetch`
  (`SharpEmu.ShaderCompiler.Vulkan/Gen5SpirvTranslator.cs` L1307-1353, L3234;
  read from SharpEmu's live working tree, since `6db095e`/`db4339f` make
  per-commit reads misleading) — build a vertex attribute's SPIR-V interface
  type as `componentKind(numberFormat) x componentCount` for **all** widths
  1..=4 and all three numeric classes (`numberFormat 4 => Uint`, `5 => Sint`,
  else `Float`), and **bitcast** raw integer components into the float-backed
  register representation rather than numerically converting them — is
  re-implemented in `kyty-graphics`' shared `vertex_input_types` resolver
  (`src/shader/spirv.rs`), consumed by `Spirv::WriteGlobalVariables`,
  `Recompile_Fetch` and the `RAEEN_VS_PASSTHROUGH` diagnostic. This closed the
  measured GTA V blocker (`invalid registers_num/input format: 2/5` = two
  components of unified format 5 = `8_UINT`). Unified-format decoding uses
  Raeen's existing classifier, itself already attributed to SharpEmu's
  `Gfx10UnifiedFormat.cs`; the Rust resolver, the diagnostics counter, the
  refusal messages, and all tests are original. Findings recorded in
  `docs/sharpemu-port/gtav-shader-inputs.md`.
  From the GTA V / UE5 boot-unblocker family (upstream `a1cbff8` PR #454,
  `db4339f` PR #650, `daaeb62` PR #406, `2764aaa` PR #542): three semantics were
  re-implemented after reading SharpEmu's current tree.
  (1) **One process-wide stack-protector guard.** `HleDataSymbols.cs`'s
  *"Keep the process data symbol and every per-thread TLS copy byte-for-byte
  identical"* and `Kernel/KernelRuntimeCompatExports.cs`'s `__stack_chk_guard`
  export (one `_stackChkGuardValue` written to both the guard object and
  `fs:0x28`) established that libkernel's `__stack_chk_guard` global and every
  thread's TCB slot must hold the same word; Raeen now serves both from
  `raeen_firmware::stack_chk_guard` (`crates/raeen-firmware/src/lib.rs`,
  `crates/raeen-runtime/src/arena.rs`). Raeen keeps its own randomized
  terminator canary and its real FSGSBASE TCB — SharpEmu's Rosetta-driven
  `mov reg, fs:[0x28]` → `xor reg, reg` code patch is deliberately **not**
  ported.
  (2) **Unreal project-relative guest paths.** `NormalizeMountRelativePath`'s
  pop-with-clamp treatment of `..` (documented there as what makes a UE title's
  `../../../`-prefixed content paths land back inside `/app0` instead of walking
  out of the game folder) replaced Raeen's stricter outright refusal in
  `combine_within_mount` (`crates/raeen-kernel/src/filesystem/mod.rs`); Raeen's
  drive-qualifier, reparse-point, and canonical-containment defenses are
  retained unchanged.
  (3) **Dinkumware `_Ctype` as a data object.** The dual data/function
  registration pattern of `HleDataSymbols` + `SysAbiExport` and the
  `LibcStdioExports.cs` table convention (384 `u16` entries over `-128..=255`,
  exported pointer at the `c == 0` slot) informed publishing `_Ctype` in Raeen's
  HLE data page from the same generator that backs `_Getpctype`
  (`crates/raeen-hle/src/libc.rs`). SharpEmu has no `_Ctype` export; the flag
  layout was already independently present in Raeen.
- **`sceKernelDlsym` fallback sweep 2026-07-28**
  (`crates/raeen-hle/src/libkernel.rs`): `DispatchKernelDynlibDlsym`
  (`src/SharpEmu.Core/Cpu/Native/DirectExecutionBackend.Imports.cs:2024`) is the
  evidence for falling back past the named module handle to a process-wide
  symbol sweep and then to emulator-provided ("runtime") symbols;
  `TryResolveRuntimeSymbolAlias` (same file, L2092) independently confirms that
  `scriptingGetMem`/`scriptingFreeMem` are allocator hooks the *runtime*
  supplies rather than guest-module exports. SharpEmu aliases them to libc
  `malloc`/`free`; Raeen follows KytyPS5's `(alignment, size)` signature
  instead, since that is the reference measured to boot the affected title.
  SharpEmu's `scriptingRealloc`/`scriptingCalloc` aliases are deliberately
  **not** adopted — their argument order is unverified and a wrong guess on a
  resize corrupts the guest heap. Raeen's load-ordered sweep and all tests are
  its own.
  Patterns and behavior are **re-implemented in idiomatic Rust with reference to
  SharpEmu's C# source**; no C# is transliterated or vendored. SharpEmu's tree
  is cloned locally into the git-ignored `reference/` directory for study only;
  it is **never vendored, compiled, or committed** into Raeen.
- **NID name catalog:** SharpEmu's `scripts/ps5_names.txt` (a public symbol-name
  list) is used as candidate input to `merge_nid_catalog`, which admits a name
  only if Raeen's own SCE-NID hash reproduces the NID from it. The result is
  factual hash data (public symbol names, no Sony code/keys), folded into
  `crates/raeen-firmware/src/dynlib/nid_names.txt`.

GPL-2.0 is the same license as Raeen (GPL-2.0-only), so derived
re-implementations are license-compatible; this notice preserves attribution.

---

## shadPS4 — PS4 emulator (reference source; NID→name data incorporated)

- **Upstream:** https://github.com/shadps4-emu/shadPS4
- **License:** GPL-2.0-or-later (`SPDX-License-Identifier: GPL-2.0-or-later`;
  the repository's `LICENSE` is the GNU GPL **Version 2, June 1991** text)
- **Copyright:** © 2024 shadPS4 Emulator Project and contributors
- **How Raeen uses it:** Primarily an Orbis HLE reference (memory, libkernel,
  linker, Vulkan), re-implemented in Rust rather than transliterated.
  The 2026-07-26 refresh to `d976c33` exposed the stale-wake failure class in a
  condition-wide generation counter: a signal intended for one waiter can be
  observed by every waiter. Raeen's FIFO/per-waiter Rust condition queue and
  tests are an original implementation informed by shadPS4 commit `26f4270`;
  no C++ is copied.

  The 2026-07-27 eliminate-fast-clear handling in
  `crates/raeen-gpu/src/draw_translate.rs` (`cb_mode`, `fast_clear_image`,
  `OffscreenDrawSink::eliminate_fast_clear`) is a Rust re-implementation of
  shadPS4's `Rasterizer::FilterDraw`/`EliminateFastClear` pattern
  (`src/video_core/renderer_vulkan/vk_rasterizer.cpp`): a CB special pass
  (CB_COLOR_CONTROL mode 2) is consumed and applied as a direct clear of the
  bound target; resolve/decompress passes are named skips. The packed
  CLEAR_WORD splat is Raeen's own simplification (shadPS4 unpacks per format
  for `vkCmdClearColorImage`); no C++ is copied.

  The 2026-07-29 PM4 decoder-agreement batch cites `pm4_opcodes.h` L33
  (`DrawIndexMultiAuto = 0x30`) in `crates/kyty-graphics/src/pm4.rs` as one
  source for the opcode number, together with the factual observation that
  `src/video_core/amdgpu/liverpool.cpp`'s packet `switch` has no case for it —
  shadPS4 names the opcode but does not walk its body. No C++ is copied.

  The 2026-07-28 Orbis exception-delivery work in
  `crates/raeen-hle/src/exception.rs` is a Rust re-implementation informed by
  shadPS4's `core/libraries/kernel/threads/exception.cpp`/`.h`: the handler ABI
  (`void handler(int signum, ucontext_t *)`), the Orbis-allowed signal set, and
  the FreeBSD amd64 `Ucontext`/`Mcontext` field offsets that Raeen's layout
  constants are pinned against by test. shadPS4 delivers on Windows through a
  special user APC (`NtQueueApcThreadEx`) rewriting the target thread's
  `PCONTEXT`; Raeen instead queues the raise and delivers at the target thread's
  next HLE safe point through its own `call_guest` re-entry, so no C++ and no
  delivery mechanism is copied.

  The 2026-07-29 extension of that work — delivering a queued exception into a
  thread that is already parked in a blocking HLE wait — takes one further fact
  from the same shadPS4 file: `sceKernelInstallExceptionHandler` installs the
  title's handler with `POSIX_SA_RESTART`, which is why Raeen **resumes** an
  interrupted wait instead of returning an `EINTR`-shaped error to the guest.
  No code.

  The 2026-07-28 `libSceAudioIn` capture library
  (`crates/raeen-hle/src/libsce_audio_in.rs`) re-implements the **values and
  contract** of `src/core/libraries/audio/audioin.cpp` / `audioin_error.h` /
  `audioin.h` in Rust: the `ORBIS_AUDIO_IN_ERROR_*` codes, the
  `ORBIS_AUDIO_IN_SILENT_STATE_DEVICE_NONE` bit reported when no microphone is
  available (`audioin.cpp:250-252`), the `(type << 16) | port_id | 0x30000000`
  handle encoding (`audioin.cpp:141`), the S16 mono/stereo `param` decode, and
  `HqOpen` routing to the same `Open` path. Raeen's port table, silence
  zero-fill, grain pacing, and tests are its own. The same date's guest-path
  normalization in `crates/raeen-kernel/src/filesystem/mod.rs` follows
  `src/core/file_sys/fs.cpp:46`'s doubled-slash correction (as a *consequence*
  of dropping empty components, not as a special case), and
  `fs.cpp:104`'s treatment of `/hostapp` as a second name for the app root is
  the reason `/hostapp` is excluded from Raeen's devkit-only root list. No C++
  is copied.

  **Data incorporated in-tree:** `crates/raeen-firmware/src/dynlib/nid_names.txt`
  is derived from shadPS4's `src/core/aerolib/aerolib.inl` — a generated table
  of public SCE symbol names and their NIDs. Raeen uses it strictly as a
  **candidate dictionary**: an entry is admitted only if Raeen's own
  `dynlib::nid::nid_of()` reproduces the NID from the name, so every retained
  name is a verified SHA-1 preimage rather than a trusted assertion (94,247 of
  aerolib's 94,276 entries pass; 29 are rejected). The test
  `nid_names::tests::all_names_hash_to_their_nid` re-proves the entire table on
  every run. Regenerate with the adjacent `gen_nid_names.py`.

  These are **public symbol names, not Sony code** — no SDK headers, firmware,
  keys, or binaries are involved, consistent with `.claude/skills/clean-room`
  ("NID names from community databases OK"). shadPS4's tree itself is cloned
  only into the git-ignored `reference/` directory and is never compiled or
  committed.

GPL-2.0-or-later may be exercised under GPL-2.0 terms, so the incorporated data
is license-compatible with Raeen's GPL-2.0-only; this notice preserves
attribution as that license requires.

---

## ps5-payload-dev/sdk — PS5 payload SDK (NID candidate source, names only)

- **Upstream:** https://github.com/ps5-payload-dev/sdk
- **License:** GPL-3.0-only (repo-wide; `include/freebsd` files are BSD) —
  **incompatible for code**: nothing from this project is compiled, linked, or
  vendored into Raeen. Its tree is cloned locally into the git-ignored
  `reference/ps5-payload-sdk` directory only.
- **How Raeen uses it:** symbol *identifiers* from its public headers were used
  as **candidates** for the NID dictionary, via `merge_nid_catalog`. A
  candidate is admitted only when Raeen's own `dynlib::nid::nid_of()`
  reproduces a real NID from it — so what lands in
  `crates/raeen-firmware/src/dynlib/nid_names.txt` is factual hash data (a
  short functional identifier plus its independently recomputed SHA-1
  preimage), not copied SDK content. Measured 2026-07-25: 35,181 of 37,345
  candidates added new hash-verified names. This is the same admission rule
  `.agents/skills/clean-room` grants for community NID databases; no SDK code,
  headers, or build files were incorporated.

## idc/ps4libdoc — PS4 library documentation (consulted; nothing incorporated)

- **Upstream:** https://github.com/idc/ps4libdoc
- **License:** none stated in the repository.
- **Measured result:** its 42,010-name `known_names.txt` was run through the
  same hash-gated merge on 2026-07-25 and added **zero** new names — the
  existing shadPS4/SharpEmu-derived catalog already covered every entry.
  Nothing from this source is incorporated; it is recorded here because it was
  evaluated as a candidate source.

## Mesa AddrLib — AMD surface-layout reference (acquired; no code incorporated yet)

- **Upstream:** https://gitlab.freedesktop.org/mesa/mesa
- **Pinned reference:** `main` at `780727e68adc`
  in git-ignored `reference/mesa`.
- **License:** the acquired `src/amd/addrlib/` files carry
  `SPDX-License-Identifier: MIT` and AMD copyright notices. The reference's
  `licenses/MIT` text is retained in the local clone.
- **How Raeen uses it:** Phase 0 establishes this as the authoritative,
  machine-pinned source for later clean-room AddrLib tiling work. No Mesa code
  or tables have been copied into Raeen in this phase. Any later transcription
  must cite the exact source file/revision and preserve its MIT attribution.
  Separately, the GTA V Phase A AGC batch (2026-07-27) uses Mesa as the
  factual reference for architectural PM4 packet identities and sizes —
  `src/amd/common/sid.h` opcodes (`PKT3_ATOMIC_MEM`, `PKT3_COND_WRITE`,
  `PKT3_PRIME_UTCL2`, `PKT3_MEM_SEMAPHORE`) and the 9-DWORD ATOMIC_MEM shape
  from `ac_cmdbuf_cp.c` — cited in `crates/raeen-hle/src/libsce_agc.rs` doc
  comments; these are hardware-interface facts, and no Mesa code was copied.
  The PM4 decoder-agreement batch (2026-07-29) cites the same file for
  `PKT3_DRAW_INDEX_MULTI_AUTO` (`sid.h` L70) in
  `crates/kyty-graphics/src/pm4.rs`, and records the *negative* finding that
  `ac_gather_context_rolls.c` only classifies that opcode as context-busy
  without decoding a body — no layout was transcribed, because none exists to
  transcribe. The thread-dimension dispatch fix (2026-07-29) additionally uses
  `src/amd/registers/gfx940.json` as the register fact that bit 5 of
  `COMPUTE_DISPATCH_INITIATOR` is `USE_THREAD_DIMENSIONS`; Raeen's Rust
  conversion and tests are original, and no Mesa code or table is copied.
  The GTA V skipped-context-register decode (2026-07-29) uses
  `src/amd/registers/gfx103.json` as the authoritative naming and bit-layout
  fact for the 50 context/user-config registers a measured frame writes and the
  PM4 decoder ignored: the `register_mappings` table resolves each context
  offset to a name via `mm = 0x28000 + offset * 4` (user-config
  `0x30000 + offset * 4`), and `register_types` supplies the field bit ranges
  for `PA_SC_MODE_CNTL_0` (MSAA_ENABLE 0, VPORT_SCISSOR_ENABLE 1,
  LINE_STIPPLE_ENABLE 2), `PA_CL_CLIP_CNTL` (VTX_KILL_OR 21,
  DX_RASTERIZATION_KILL 22) and `CB_SHADER_MASK`. These are
  hardware-interface facts used to name registers and place bits; the Rust
  tables, decode and tests in `crates/kyty-graphics/src/pm4.rs` and
  `crates/kyty-graphics/src/run.rs` are original, and no Mesa code, JSON or
  generated header was copied into the tree.

---

## Compiled Rust crate dependencies

Unlike the clean-room reference sources above (studied but never linked),
these crates.io dependencies are compiled into Raeen (or its test binaries).
Only licenses compatible with GPL-2.0-only are used.

- **iced-x86** — https://github.com/icedland/iced — MIT, used by the module
  linker to identify real x86-64 `syscall` instructions in executable guest
  segments. Those instructions are trapped into the Orbis syscall dispatcher
  so a PS5 syscall number can never be executed against the Windows kernel.

- **rspirv** — https://github.com/gfx-rs/rspirv — dual MIT / Apache-2.0, used
  here under its **MIT** option (Apache-2.0 is *not* GPLv2-linking-compatible;
  MIT is). **Test-only** dev-dependency of `raeen-gpu`: it structurally
  validates the shader emitter's SPIR-V output in unit tests and is **not
  linked into the distributed emulator binary**.

- **naga** — https://github.com/gfx-rs/wgpu (naga crate) — dual MIT /
  Apache-2.0, used here under its **MIT** option. **Test-only** dev-dependency
  of `kyty-graphics` (`spv-in` feature): its SPIR-V front end parses the
  binaries produced by the `spirv_asm` assembler in unit tests as an extra
  validity gate. It is **not linked into the distributed emulator binary**
  through this use (naga also ships transitively inside the GUI's wgpu stack,
  which is an unrelated, already-present dependency).

---

## Algorithms implemented from published descriptions (no code copied)

- **AMD FidelityFX Super Resolution 1.0 (FSR1)** — MIT (GPL-2.0 compatible).
  `raeen-upscale`'s `fsr` backend (`spatial::fsr1`) implements FSR1's two-pass
  *approach* — EASU (edge-adaptive spatial upsampling) followed by RCAS
  (robust contrast-adaptive sharpening) — written from the published
  description of the algorithm. **AMD's shader source was not copied or
  ported**, so the result is FSR1-*class* and deliberately not bit-identical
  to AMD's implementation. FSR1 is spatial-only: no motion vectors, no vendor
  runtime, which is why it can ship in-tree at all where DLSS and XeSS cannot.

- **Host sleep strategy (Windows timer precision)** — `raeen-core`'s
  `host_sleep` module. The *mechanisms* of four reference emulators' guest-sleep
  paths were read and compared before it was written, and none of their code was
  copied: kytyps5 `src/common/threads.cpp` (thread-local
  `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` with a QPC spin below 1 ms, GPL-2.0),
  shadPS4 `src/common/thread.cpp` `AccurateSleep` (per-call plain
  `CreateWaitableTimer`, GPL-2.0-or-later with Dolphin/Citra lineage), SharpEmu
  `src/SharpEmu.Libs/HostTiming.cs` (a four-tier sleep/yield/spin ladder with a
  100 µs spin threshold, GPL-2.0-or-later), and Kyty
  `source/lib/Core/src/Threads.cpp` (plain `sleep_for`, MIT). Raeen's
  implementation diverges from all four on measured grounds — it uses a
  `PAUSE`-only spin with an admission limit, where kytyps5 and SharpEmu yield
  inside their spin, because a yielding spin measured 60–190 ms per sub-millisecond
  request on an oversubscribed host. The thread-local cached-timer shape it
  generalises already existed in-tree in `raeen-hle/src/libsce_video_out.rs`.
  Licences are recorded here because the designs were *studied*; no lines were
  taken from any of them.

---

## Not incorporated (ecosystem references only)

The following projects were evaluated. Their **code is not used** in Raeen —
they are GPL-3.0, which is incompatible with this project's GPL-2.0-only
license, and they target real (jailbroken) PS5 hardware rather than emulation.
They are noted only as ecosystem references (e.g. for the homebrew payload
format and the set of `sceKernel*` calls real homebrew invokes):

- **cy33hc/ps5-payload-loader** — GPL-3.0 — on-console homebrew payload loader.
- **phantomptr/ps5upload** — GPL-3.0 — desktop → console file-transfer tool.

If any of their code were ever to be incorporated, Raeen would first have to
move to GPL-3.0(-or-later); that has not been done.
