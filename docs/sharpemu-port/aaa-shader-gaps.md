# Three AAA shader gaps — ASTRO.BOT, GTA V, Avatar

**Date:** 2026-07-29
**Branch:** `fix-aaa-shader-gaps`
**Crates touched:** `kyty-graphics`, `raeen-gpu`
**Evidence:** `artifacts/compat/post-astrofix-validated.json` (build `1acd114`)
plus a fresh dump-and-disassemble pass over the retail titles on this machine
(2026-07-29, `RAEEN_DUMP_SHADERS` → `tests/dump_disasm.rs`).

This is a decode / translate / binding claim with in-tree tests. **No title was
re-measured to a stage change**, so nothing here is a rendering claim for any of
the three. Avatar *was* re-run twice and its blocker measurably advanced through
two distinct defects to a third, correctly-named one — that progression is
evidence the diagnoses are right, not that the title renders.

---

## 1. ASTRO.BOT — `parse: unknown sop2 opcode: 0x30`

### Identity

SOP2 `0x30` is **`s_lshl3_add_u32`**: `sdst = (ssrc0 << 3) + ssrc1`, with SCC
set from the 33-bit carry-out.

Three independent sources agree row-for-row across the whole family, so the
identity is established rather than inferred:

| Source | Location | Rows |
|--------|----------|------|
| **LLVM AMDGPU backend** (gfx10 = RDNA/RDNA2) | `llvm/lib/Target/AMDGPU/SOPInstructions.td` | `SOP2_Real_gfx10<0x02e>` `S_LSHL1_ADD_U32`, `<0x02f>` `S_LSHL2_ADD_U32`, **`<0x030>` `S_LSHL3_ADD_U32`**, `<0x031>` `S_LSHL4_ADD_U32`, `<0x032..0x034>` `S_PACK_LL/LH/HH_B32_B16`, `<0x035>` `S_MUL_HI_U32`, `<0x036>` `S_MUL_HI_I32` |
| KytyPS5 (MIT) | `src/graphics/shader/recompiler/ScalarAluOps.cpp` L26-28 | identical for 0x2e-0x31 |
| SharpEmu (GPL-2.0) | `src/SharpEmu.ShaderCompiler/Gen5ShaderTranslator.cs` L804-807 | identical for 0x2e-0x36 |

**On the vendor ISA doc.** The AMD RDNA 2 ISA Reference Guide (doc 70648) is the
intended primary source, but it could not be fetched from this environment:
`docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture` serves only a
metadata page, and the underlying
`amd.com/content/dam/.../rdna2-shader-instruction-set-architecture.pdf` returned
a timeout and then `ECONNRESET`. LLVM's AMDGPU backend was used in its place as
the highest-precedence available source: it is the machine-readable gfx10
encoding table AMD co-develops, and it agrees exactly with both emulator
sources. The identity for `0x30` is therefore not in doubt; a later pass with
working access to the PDF should still confirm the SCC wording.

**One divergence worth recording.** LLVM's gfx10 section does **not** list
`S_ABSDIFF_I32` at all (it appears only from gfx11). This tree carries a *named
not-implemented* refusal for `s_absdiff_i32` at `0x2c` (GFX9's numbering) while
SharpEmu puts it at `0x2d`. Neither is confirmed for RDNA2. Both are unreached
NI paths today, so nothing was changed — but if a title ever hits `0x2c`/`0x2d`,
the current label should not be trusted.

Semantics from SharpEmu `Gen5ShaderScalarEvaluator.cs` L1525-1552 (one body for
all four shifts: `wide = (left << N) + right`, result = low 32 bits, SCC =
`wide > uint.MaxValue`) and the SPIR-V lowering from
`Gen5SpirvTranslator.Alu.cs` L2122-2132 (`IAdd(ShiftLeftLogical(left, N), right)`).

### The neighbourhood

`0x31` (`s_lshl4_add_u32`) was already fully wired — parse, dispatch row, SPIR-V
`%lshl_add` helper, scalar fold. `0x2e`, `0x2f` and `0x30` all fell through to
`unknown_op`. A missing opcode is rarely alone, and here the three siblings are
literally the same instruction with a different `N`, so all three were decoded
together rather than waiting for each to be measured.

The rest of the immediate neighbourhood was checked and left alone: `0x2c`
(`s_absdiff_i32`), `0x33`/`0x34` (`s_pack_lh`/`hh`) already carry *named*
not-implemented refusals, which is the correct state — they are identified, just
not lowered.

### What now translates

* `parse.rs` — `0x2e..=0x31` decode as one arm behind the existing `next_gen`
  gate, into new `SLshl1AddU32` / `SLshl2AddU32` / `SLshl3AddU32`.
* `recompile.rs` — three dispatch rows on `SVdstSVsrc0SVsrc1`, each calling the
  existing `%lshl_add` helper with its own shift constant and `SccCheck::CarryOut`.
* `spirv.rs` — `FUNC_LSHL_ADD` emission is gated on all four types, not just `0x31`.
* `scalar_eval.rs` — all four fold, so a byte offset computed with any of them
  resolves for the V#/pointer capture passes instead of leaving them refusing.

`dispatch_table_counts`: **364 → 367** rows, **356 → 359** implemented, `ni`
unchanged at 8.

---

## 2. GTA V — `SBufferLoadDword ... x1 ... no resolved capture`

### What the measurement said

```
Recompile_SBufferLoadDword_SdstSvSoffset: not supported: no storage buffer bound
for the V# and no resolved capture: SBufferLoadDword [SdstSvSoffset] x1 dwords,
V#=s[16:19], soffset=none, imm=0x90, pc=0x90     (1173 occurrences)
                                    V#=s[12:15], imm=0x90, pc=0x88   (45)
SBufferLoadDwordx16 ... V#=s[0:3], soffset=none, imm=0x0, pc=0xdc    (452)
```

The x1 path **is** routed through `sbuffer_load_dwords` and therefore through
`shader_capture_vsharp_buffer_loads`. So this is a rejected provenance, not a
missing route.

### The rejected provenance

The capture accepted exactly one producer shape:

```rust
if producer.dst.register_id != base_reg || producer.dst.size < 4 { continue; }
```

That proves only the **first** quad of a descriptor table. A single wide scalar
load delivers several V#s at once — and the retail disassembly shows this is the
normal shape, not an edge case:

* Avatar VS `0x4020024d00` (dumped 2026-07-29):
  `s_load_dwordx16 s[12:27], s[8:9], 0` → **four** V#s at s[12:15], s[16:19],
  s[20:23], s[24:27], all four used as MUBUF bases in the next five instructions.
* GTA V logs the shape directly: `SLoadDwordx8 [Sdst8SbaseSoffset] s[8:15], s[0:1], 0`
  — a two-V# table whose second quad is s[12:15].

The refused registers in the measurement (`s[12:15]`, `s[16:19]`) are exactly
what a table load at a nonzero quad offset produces.

### Fix

`proved_vsharp_quad` (extracted from the capture so the SMEM and MUBUF paths
share one proof) accepts any quad **wholly inside** the producer's destination
range and reads the already-captured snapshot at that dword offset:

```rust
let quad_offset = base_reg - producer.dst.register_id;
if quad_offset < 0 || quad_offset + 4 > producer.dst.size { return None; }
let end = at_dword + 4;
if end > captured.dwords_num as usize { return None; }   // only PROVED dwords
```

No address is guessed: these are the same bytes the capture already proved, read
at the correct index. A partial overlap, an offset past `dwords_num`, more than
one writer, a quad assembled by `s_mov`s, or a producer whose own dwords were
never captured all still return `None` and keep the named refusal.

### Honest limit

The relaxation itself is **confirmed working against a retail title** — just not
this one. Avatar's four-V# table load exercises exactly this path (three of its
four quads are at nonzero offsets), and the measured run shows those quads
resolving: they now bind and reach the format check. So the mechanism is proven
end-to-end; only its effect on GTA V specifically is unverified.

GTA V's specific failing shader (`ps 0x148d84000`) could **not** be re-captured
on this machine: both attempts stalled early in `sceAgcDcbAcquireMem` and the
title never reached that draw (11 flips vs. the baseline's 256). The shaders that
were captured (`ps_148d83000`, `cs_148d8d000`) show the SRT V# idiom
(`s_load_dwordx4 s[32:35], s[28:29], 0x20` → `s_buffer_load_dword vcc_lo,
s[32:35], 0x10`) and both translate cleanly today. So the offset-quad gap is
established from Avatar's dumped table load and from GTA's own logged
`s[8:15]` producer, **not** from re-running the exact failing shader. Whether it
is GTA V's whole x1 blocker is for the next baseline to say.

---

## 3. Avatar: Frontiers of Pandora — `can't recompile: BufferLoadFormatXyzw`

### Precise diagnosis

The brief's hypothesis was that the row exists and the non-119 unified-format
path leaves the destination untouched. **That is not what Avatar hits**, though
it is a real latent bug (see below).

`recompile_buffer_load_format_xyzw_vdata4` is structured as:

```rust
if let Some(bind_info) = spirv.get_bind_info() {
    if bind_info.storage_buffers.buffers_num > 0 {
        ... // the descriptor-format branch lives HERE
        return Ok(true);
    }
}
Ok(false)          // <- the measured path; caller prints a bare "can't recompile:"
```

So: the row exists, the recompiler is implemented, and it returns `Ok(false)`
because `storage_buffers.buffers_num == 0`. The format branch is inside the
`buffers_num > 0` arm and was never reached. The bare `can't recompile:` in the
log — with no parenthesised reason — is precisely the signature of `Ok(false)`.

Reproduced on this machine 2026-07-29: 853 occurrences in a 180 s run, all
`v[0:3], v5, s[12:15], 0, idxen`.

### Why nothing was bound

Dumped VS `0x4020024d00` (36 instructions) is a plain five-stream vertex fetch:

```text
s_load_dwordx16         s[12:27], s[8:9],   0     ; table of four V#s
s_load_dwordx4          s[0:3],   s[8:9],   0x40  ; a fifth
buffer_load_format_xyzw v[0:3],   v5, s[12:15], 0 idxen
buffer_load_format_xyzw v[6:9],   v5, s[0:3],   0 idxen
buffer_load_format_xyzw v[10:13], v5, s[24:27], 0 idxen
buffer_load_format_xyzw v[14:17], v5, s[20:23], 0 idxen
buffer_load_format_xyzw v[18:21], v5, s[16:19], 0 idxen
```

`s[8:9]` is live-in user data; the V#s arrive through an SRT table the
usage-table walk never turns into a bound slot.

A MUBUF `idxen` fetch **cannot** be snapshotted the way
`shader_capture_vsharp_buffer_loads` snapshots a scalar load — its address
depends on the per-invocation index in `v5`. The correct lowering is the one the
recompiler already has: bind the descriptor and read live guest memory at draw
time.

### Fix — `shader_bind_vsharp_storage_buffers`

A new analysis pass binds a MUBUF/MTBUF V# as a real (non-extended) storage
buffer when — and only when — `proved_vsharp_quad` proves it. A V# the usage
table already bound is never shadowed. Chained from
`shader_capture_runtime_scalar_loads_shifted` (VS/PS) and called directly for CS
in `raeen-gpu::shader_fetch`, mirroring how the SMEM capture is wired.

### The device-loss hazard the binding creates, and its guard

`Spirv::WriteLocalVariables` seeds a non-extended binding's four SGPRs from the
push-constant window, in which **dword 0 has been rewritten** from the guest base
address to the compact descriptor-array index, and every buffer body indexes
`%buf` with the value of that register. `write_local_variables` runs *before*
`write_instructions`, so the captured `s_load_dwordx16` would then materialize
the **raw guest dwords over the seed** — turning a small array index into a guest
address and indexing the descriptor array out of bounds. That is exactly the
shape that produced a measured `VK_ERROR_DEVICE_LOST` on ASTRO.BOT
(`mimg_descriptor_guard`, shape 2).

`sload_dword_extended` and `sbuffer_load_dwords` now skip any captured dword
whose destination register a non-extended **storage** binding owns
(`descriptor_seeded_register` → `storage_binding_owns_register`). This is the
correct lowering, not a workaround: the seed is the value the shader must
observe, and the descriptor's real content still reaches the draw, live, through
the binding. The guard is deliberately scoped to storage bindings so no existing
texture/sampler behaviour changes.

### All or nothing

`mubuf_flexible` has two lowerings. With **no** storage buffer bound anywhere it
treats every MUBUF as a null V# — loads return 0, stores drop — and the shader
still compiles. With **at least one** bound it switches every site to the
descriptor path, which indexes `%buf` with the value of that site's V# dword-0
register. A site whose V# was not proved has no seeded register, so it would
index the descriptor array with raw guest data.

Binding some-but-not-all of a shader's V#s would therefore convert a compiling
(if zero-valued) shader into a device-loss risk. The pass pre-scans every
MUBUF/MTBUF site and binds nothing unless all of them end up covered.

### The second defect, found by re-measuring: a unit mismatch

Binding the V# moved Avatar's blocker, and the new one named itself:

```
V# unified format 77 is not 119 (32_32_32_32_FLOAT):
BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen] V#=s[12:15], stride=32, pc=0x68
                                                              (985 occurrences)
```

**Unified 77 *is* `32_32_32_32_FLOAT`.** The two numbers are the same format in
two different unit systems:

* SharpEmu `Gfx10UnifiedFormat.cs` L89 (RDNA2 ISA table 47): unified
  `77 => (dfmt 14, nfmt 7)`. The table has **no entry 119** at all.
* Kyty's helper comments give the packing: `tbuffer_load_format_xyzw` tests
  `dfmt_nfmt == 119 // dfmt = 14, nfmt = 7`, i.e. `dfmt * 8 + nfmt`. All five of
  Kyty's constants confirm it — 36 = (4,4), 39 = (4,7), 92 = (11,4),
  95 = (11,7), 119 = (14,7).

MTBUF carries `dfmt`/`nfmt` in the **instruction**, so Kyty's MTBUF rows hardcode
the packed number (`OpStore %temp_int_5 %int_36`). MUBUF takes the format from
the **descriptor**, where RDNA2 stores the *unified* number — and the MUBUF row
extracted it at runtime and passed it straight in:

```
%t208 = OpShiftRightLogical %uint %t206 %int_12
%t210 = OpBitwiseAnd %uint %t208 %uint_127     ; unified format
        OpStore %temp_int_5 ...                ; -> compared against 119
```

That comparison **can never succeed for any descriptor**, so
`buffer_load_format_xyzw` silently left its destination VGPRs untouched every
single time. This is the defect the handoff note suspected, one layer deeper
than expected: not "non-119 formats are unsupported" but "the row was speaking
the wrong dialect".

Fixed by converting at translate time from the bound descriptor
(`gfx10_unified_to_packed_dfmt_nfmt`, the table ported from SharpEmu with
attribution) and storing the packed constant, replacing the runtime extraction
entirely. Avatar's unified 77 now reaches the helper as 119 and the fetch fires.

A format that genuinely has no packed spelling the helper serves is a **named,
counted refusal** (`unsupported_buffer_format_skips`, reported by
`record_shader_skips`) giving the unified number, its decoded `dfmt`/`nfmt`, the
V# registers, the stride and the destination VGPRs. A V# that is not one of the
shader's bound descriptors is likewise refused by name rather than guessed at.

### Measured progression (three runs of the retail title, this machine)

Each fix moved the blocker to the next real thing, and each new message named
itself:

| Build | Top shader refusal | Count / 180 s |
|-------|--------------------|--------------:|
| `40f3970` (baseline) | `can't recompile: BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen] v[0:3], v5, s[12:15], 0, idxen` — bare `Ok(false)`, no descriptor bound | 853 |
| + V# binding + offset quad | `V# unified format 77 is not 119 (32_32_32_32_FLOAT) ... V#=s[12:15], stride=32` — descriptor bound, wrong units | 985 |
| + unit conversion | `V# unified format 56 (dfmt 10, nfmt 0) is not 32_32_32_32_FLOAT (unified 77 / packed 119)` | 769 |

Unified 77 refusals are now **zero** — those streams translate. What remains is
a genuine capability gap, finally stated in the right units: **dfmt 10 = 8_8_8_8,
nfmt 0 = UNORM**, i.e. Avatar packs vertex attributes as four normalized bytes,
which Kyty's 32-bit-float-only helper does not implement. An `8_8_8_8_UNORM`
unpack (load one dword, extract four bytes, scale by 1/255) is the next step and
is a real feature, not a bug fix — so it was not invented here.

`Ok(false)` in the same recompiler was also replaced with a named refusal, so a
future `buffers_num == 0` case says so instead of printing a bare
`can't recompile:`.

---

## Tests

All spirv-val (Vulkan 1.3) validated; naga is not the gate here (documented
false negatives in this crate).

| Test | File | Gate |
|------|------|------|
| `s_lshl_n_add_u32_decodes_every_shift_and_lowers_through_lshl_add` | `shader/recompile.rs` | encode-bytes → decoded instruction for all four SOP2 opcodes, row + `CarryOut` SCC rule, `%lshl_add` called with the right shift constant, spirv-val |
| `every_s_lshl_n_add_u32_shift_folds` | `shader/scalar_eval.rs` | all four shifts fold; low-32 result with the carry going to SCC |
| `full_chain_vsharp_at_an_offset_inside_a_loaded_table` | `shader/recompile.rs` | real SMEM bytes → `s_load_dwordx8 s[16:23]` table → `s_buffer_load_dwordx4 s[20:23]` (second quad) → capture → recompile → spirv-val |
| `avatar_srt_vsharp_binds_and_its_descriptor_seed_survives` | `shader/recompile.rs` | Avatar's dumped shape: `s_load_dwordx16 s[12:27]` + `buffer_load_format_xyzw ... s[16:19] idxen`; asserts the binding is created, and that s16..s19 keep **exactly one** store (the seed) while unowned table dwords still materialize |
| `a_shader_with_one_unprovable_mubuf_vsharp_binds_none_of_them` | `shader/recompile.rs` | the all-or-nothing rule below |
| `unified_format_converts_to_the_packed_number_kyty_helpers_test` | `shader/recompile.rs` | all five of Kyty's documented packed constants round-trip from their unified encodings; unified 119 does not exist; reserved holes and image-only encodings stay `None` |
| `non_119_buffer_format_is_refused_by_name_not_silently_dropped` | `shader/recompile.rs` | a bound unified-13 descriptor refuses by name (with its decoded dfmt/nfmt) and increments the counter |

Failing-first was verified for the two provenance tests by restoring the
`quad_offset == 0` restriction: both fail, at the same site.

Counts: kyty-graphics **595 lib** (from 588; seven new tests) plus its integration
suites unchanged at 8 (`gcn_to_spirv` 5, `diagnose_terrain_atlas` /
`dump_disasm` / `enumerate_dumps` 1 each — the handoff note's "6" appears
stale), raeen-gpu **310 lib**. `cargo test --workspace` green.
`cargo fmt --all` clean; `cargo clippy --workspace --all-targets -- -D warnings`
exits 0.

---

## The same unit mismatch in the four other rows — now closed

`BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen]` was fixed first because it is
the measured blocker. The identical runtime extraction
(`OpShiftRightLogical %int_12` + `OpBitwiseAnd %uint_127` → a helper that tests
the *packed* number) survived at `recompile.rs` ~785, ~3749, ~3833 and ~4058 —
the `BufferLoadFormatX` / `BufferStoreFormatX` / `BufferStoreFormatXy` bodies
and `mubuf_flexible`'s shared path — so those MUBUF typed accesses were
**silently no-ops for every descriptor**, exactly as `Xyzw` was. (The follow-up
note said `BufferStoreFormatXyzw` for the third site; the standalone row there
is `BufferStoreFormatXy`. `BufferStoreFormatXyzw` reaches the defect through
`mubuf_flexible`, the fourth site.)

All four now resolve the format at translate time. The conversion is factored
into one resolver rather than copied five ways:

* `TypedBufferHelper` describes a Kyty typed helper by what its own `OpIEqual`
  guard accepts — `tbuffer_load_format_x` / `tbuffer_store_format_x` 36 & 39,
  `tbuffer_store_format_xy` 92 & 95, `tbuffer_{load,store}_format_xyzw` 119 —
  taken from the constants in `spirv.rs`, not restated from the ISA.
* `mubuf_descriptor_packed_format` finds the instruction's V# among
  `bind.storage_buffers` (`start_register[i] + shift == inst.src[1].register_id`,
  shift 8 for a gs-prolog VS), converts `buffers[i].format()` through
  `gfx10_unified_to_packed_dfmt_nfmt`, and returns the packed constant the row
  then emits as `OpStore %temp_int_5 %int_<packed>`. All five packed constants
  are already declared by `find_constants`, so no new SPIR-V id is needed.
* A format the row's helpers do not serve is refused **by name** — naming both
  numbering schemes, since the fix is descriptor-side — and counted in
  `UNSUPPORTED_BUFFER_FORMAT_SKIPS`.

`gfx10_packed_to_unified_dfmt_nfmt` (spirv.rs) inverts the table so a refusal
can print `32_32_32_32_FLOAT (unified 77 / packed 119)` without a second
transcription of table 47.

**Behaviour change worth stating.** These rows previously always *compiled*;
they just did nothing. They can now fail the shader. That is the same trade the
`Xyzw` row already made — a silent no-op is invisible in a log and a refusal is
not — and the skip counter is what measures how much a real format unpack would
recover. The four-channel load is the case where that follow-up landed: it now
carries two candidates (`tbuffer_load_format_xyzw` for 119 and
`buffer_load_format_xyzw_unorm8` for 80), and the resolver picks between them
from the bound descriptor.

`mubuf_flexible` is shared by ten format-carrying dispatch rows plus the raw
dword/ubyte rows, so each is covered separately:
`every_mubuf_typed_row_passes_the_packed_format_not_the_unified_one` drives all
fourteen typed rows (four standalone + ten flexible) from real MUBUF encodings
through to spirv-val-clean SPIR-V, asserting both the packed constant and the
absence of the runtime extraction;
`a_format_the_helper_cannot_serve_is_refused_per_row_and_counted` checks the
named refusal per row; `the_raw_mubuf_rows_keep_no_format_argument` proves the
raw rows gained no format lookup and therefore no new refusal.

Counts after this pass: kyty-graphics **601 lib**, raeen-gpu **310**.

## Avatar's next named gap

Visible in the same run, behind the `Xyzw` one:

```
can't recompile (no table entry for BufferLoadFormatXyz/Vdata3VaddrSvSoffsIdxen):
BufferLoadFormatXyz [Vdata3VaddrSvSoffsIdxen] v[14:16], v5, s[...], 0, idxen
```

This one really is a missing row — and unlike `Xyzw` it has no existing
three-channel typed helper to compose from (`tbuffer_load_format_xyz` does not
exist upstream). Left for a follow-up rather than invented here.

## Not claimed

* **No title reached a new stage.** ASTRO was not re-run at all; GTA V stalled
  before its failing draw on this machine (`sceAgcDcbAcquireMem`, 11 flips vs.
  the baseline's 256); Avatar's blocker was reproduced and measurably advanced
  twice, but a shader that still refuses at one site still fails, so this is a
  translate-path result and not a rendering claim.
* Avatar's `32_32_32_32_FLOAT` streams now translate; its `8_8_8_8_UNORM`
  streams do not, and that refusal fails the shader. The title is closer, not
  working.
* The offset-quad relaxation is established from Avatar's dumped table load and
  GTA V's logged `s[8:15]` producer, not from re-running GTA V's exact failing
  pixel shader.

---

# Follow-up (2026-07-28): Avatar's `8_8_8_8_UNORM` unpack

Closes the capability gap §3 left open ("An `8_8_8_8_UNORM` unpack … is the next
step and is a real feature, not a bug fix — so it was not invented here").

## Channel order and the UNORM rule — established, not guessed

Two references agree row-for-row, so the layout is sourced rather than inferred:

| | KytyPS5 (MIT) | SharpEmu (GPL-2.0) |
|---|---|---|
| file | `src/graphics/shader/recompiler/BufferFormat.h`, `spirvEmitter/spirvEmitterMemory.cpp` | `src/SharpEmu.ShaderCompiler.Vulkan/Gen5SpirvTranslator.cs` |
| layout | `GetFormatInfo(k8_8_8_8UNorm)` → `component_bits {8,8,8,8}`, `component_bit_offset {0,8,16,24}`, `packed_bitfield = false` | `LoadGfx10BufferFormatComponent`: `SetLayout(10, c, 0, 8)` for c = 0..3 |
| component `c` | byte at `base + c`, loaded by `EmitMemoryLoadSubDwordValueU32` (L541-575) as `(dword[a >> 2] >> ((a & 3) * 8)) & 0xff` | byte offset `c`, bit offset 0, 8 bits |
| UNORM | `NormalizeFormatComponent` (L899-908): `OpConvertUToF` then `OpFDiv` by `(1 << bits) - 1` = **255.0** | `ConvertGfx10BufferComponent`: `ConvertUToF(raw) / ConvertUToF(lowMask)`, `lowMask = 255` |

So **x is the lowest byte** of the containing little-endian dword — bits
`c*8 .. c*8+7` — and each channel is `float(byte) / 255.0`. Both sources are
cited by file and line in the helper's doc comment.

**Per-component byte addressing, not one dword load with fixed offsets.** Both
references address each component by byte (`base + c`, dword `= a >> 2`, shift
`= (a & 3) * 8`) rather than bit-slicing a single dword at 0/8/16/24. That is
what makes an element whose byte address is not 4-aligned decode correctly — it
straddles two dwords, and each byte is fetched from whichever dword holds it.
For a 4-aligned element the four extractions collapse to exactly 0/8/16/24 of
one dword, so the measured case costs nothing. It reuses the `BUFFER_LOAD_UBYTE`
pattern already in this tree.

## Shape

* `spirv.rs` — `BUFFER_LOAD_FORMAT_XYZW_UNORM8` on the existing
  `%function_buffer_load_float4` signature. **No `dfmt_nfmt` parameter and no
  `OpIEqual` guard**: the descriptor is known at translate time, so unlike the
  `tbuffer_*` helpers there is nothing left to test per invocation. Plus
  `add_constant_float(255.0)` and an emission gate on `BufferLoadFormatXyzw`.
* `recompile.rs` — `mubuf_descriptor_packed_format` now takes a slice of
  candidate helpers and returns which one the descriptor selected. The
  four-channel load passes `[&TBUF_LOAD_FORMAT_XYZW, &BUF_LOAD_FORMAT_XYZW_UNORM8]`;
  the other four rows pass a one-element slice and are otherwise unchanged. One
  descriptor lookup, one skip-counter increment, and a refusal that names every
  format the row could have served. A new `takes_format_arg` field decides
  whether the emitting row stores and passes `%temp_int_5`, so emission is
  data-driven rather than a `match` that could drift from the candidate list.

## Measured (retail, this machine, 2026-07-28)

Build `21:39:09` release, `RAEEN_VBLANK_HZ=0 RAEEN_ASYNC_FLIP=1
RAEEN_TIME_WORKER=1`, `--run-eboot`, 3 m 19 s, 71 flips, no device loss.

| | before (§3, build `+ unit conversion`) | after |
|---|---:|---:|
| `V# unified format 56` refusals | 769 | **0** |
| refusals naming `BufferLoadFormatXyzw` | 769 | **0** |
| total shader `not supported:` in the run | — | **5** |

The 51 `BufferLoadFormatXyzw` lines still in the log are plain disassembly of
sites that translated. The whole run's remaining `not supported:` volume is 5
occurrences of a **different** row — `Recompile_BufferStoreFormatXyzw_Vdata4VaddrSvSoffs`,
`unified format 75 (dfmt 14, nfmt 4)` = `32_32_32_32_UINT`, a store, out of
scope here.

The top gap is now the one §"Avatar's next named gap" predicted:
`BufferLoadFormatXyz [Vdata3VaddrSvSoffsIdxen]`, 49 sites — a genuinely missing
three-channel row with no upstream helper to compose from.

## Tests

`kyty-graphics` 601 lib (three new) + 8 integration; `raeen-gpu` 310 lib;
`cargo fmt --all --check` clean; `cargo clippy -p kyty-graphics -p raeen-gpu
--all-targets -- -D warnings` exits 0.

| Test | Gate |
|------|------|
| `a_unified_56_descriptor_selects_the_8_8_8_8_unorm_unpack` | a unified-56 descriptor calls `%buffer_load_format_xyzw_unorm8` and passes it **no** format argument; no `OpStore %temp_int_5`; no runtime `(dword3 >> 12) & 0x7f` extraction; component `c` reads byte `base + c`; four `OpBitFieldUExtract`, four `OpFDiv`, all by `%float_255_000000`; spirv-val (Vulkan 1.3); and unified 77 still takes the packed-119 float4 path |
| `the_four_channel_load_still_refuses_a_format_neither_helper_serves` | unified 13 refuses by name, listing **both** candidates and both served formats, with the load consequence wording, and is counted |
| `the_8_8_8_8_unorm_encoding_round_trips_between_both_numberings` | 56 ↔ (10,0) ↔ packed 80, and unified 80 is a different format from packed 80 |

Failing-first is established by construction rather than by hand: the
pre-existing `a_format_the_helper_cannot_serve_is_refused_per_row_and_counted`
listed `0xE00C_2000` + unified 56 as a **refusal** row, so it failed until that
row was removed and replaced with unified 13.

## Not claimed

* **The title still does not render.** `BufferLoadFormatXyz` refuses at 49 sites
  and a shader that refuses anywhere still fails. This is a translate-path
  result, not a rendering claim.
* No before/after A/B was run on this machine: the pre-change release binary was
  already overwritten, so "769" is quoted from §3's measurement, not re-measured
  today. The after-run's zero is direct, and the path is proven exercised by the
  51 translated sites and the moved blocker.
