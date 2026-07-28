# V#-based buffer loads (`s_buffer_load*` + MUBUF `idxen`)

Date: 2026-07-28
Baseline measured: `artifacts/compat/post-smem-validated-20260728.json`
(build revision `36a9b18ccbb1`)

## What was measured

Three of nine titles had a first blocker in one instruction family:

| Title | Flips | Shader errors | First blocker |
|-------|------:|--------------:|---------------|
| Grand Theft Auto V | 192 | 98 | `spirv: can't recompile: SBufferLoadDwordx8 [Sdst8SvSoffset] s[24:31], s[20:23], 0` |
| ASTRO.BOT | 128 | 126 | `parse: not implemented smem feature: offset != 0 with register soffset on an s_buffer_load (V# base)` |
| Avatar: Frontiers of Pandora | 192 | 2398 | `spirv: can't recompile (no table entry for BufferLoadFormatXyzw/Vdata4VaddrSvSoffsIdxen): BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen] v[0:3], v5, s[12:15], 0, idxen` |

Three *different* failure modes, one family:

* ASTRO died in the **parser** — no module was produced at all.
* GTA V reached the **recompiler**, whose dispatch row existed, but every
  `recompile_sbuffer_load_*` returned `Ok(false)` because
  `bind.storage_buffers.buffers_num == 0`.
* Avatar had **no dispatch row** for the (type, format) pair.

## The V# addressing rule, and where it comes from

The previous pass (`smem-register-soffset.md`) implemented the *pointer* form
`s_load*` (`base = SGPR[n:n+1]`) and deliberately refused the *descriptor* form
`s_buffer_load*` (`base = V# in SGPR[n:n+3]`) because it could not establish the
semantics. It is now established:

```text
s_buffer_load*:  addr = V#.base48 + zext(SGPR[soffset]) + sext21(imm)
                 addr &= !3
                 dword index = addr >> 2
size (BOUND only): stride == 0 ? num_records : stride * num_records
```

**The V# contributes only its base address to the address arithmetic.**
`stride` and `num_records` bound the access; they are not address terms.

### Source 1 — SharpEmu `Gen5ShaderScalarEvaluator.cs::TryExecuteScalarLoad` (L1875-1902)

Decisive, because `s_load*` and `s_buffer_load*` run through **one body** in
which only the base differs:

```csharp
var isBufferLoad = instruction.Opcode.StartsWith("SBufferLoad", ...);
var hasBufferDescriptor = isBufferLoad && TryDecodeBufferDescriptor(...);
var baseAddress = hasBufferDescriptor
    ? bufferDescriptor.BaseAddress
    : scalarRegisters[scalarBase.Value] | ((ulong)scalarRegisters[scalarBase.Value + 1] << 32);
var dynamicOffset = control.DynamicOffsetRegister is { } r ? scalarRegisters[r] : 0;
var immediateOffset = (ulong)(long)control.ImmediateOffsetBytes;
var byteOffset = unchecked(immediateOffset + dynamicOffset);
var address = unchecked(baseAddress + byteOffset) & ~3UL;
```

`byteOffset` and the `& ~3UL` truncation are **shared** between the two
families. That is exactly the fact the previous pass was missing: the combined
`soffset + immediate` rule transfers unchanged from the pointer form; only the
base is read differently.

### Source 2 — SharpEmu `TryDecodeBufferDescriptor` (L2163-2216)

What the other V# fields are for:

```csharp
var baseAddress   = word0 | ((ulong)(word1 & 0xFFFFu) << 32);
var stride        = (word1 >> 16) & 0x3FFFu;
var unifiedFormat = (word3 >> 12) & 0x7Fu;
var sizeBytes     = stride == 0 ? word2 : (ulong)stride * word2;
if ((word3 >> 30) != 0) { /* not a buffer sharp */ }
```

`stride`/`num_records` produce a **size**, never an address term. `word3 >> 30`
is the type check (Raeen's `sharp_dword3_is_buffer`), and `(word3 >> 12) & 0x7F`
is the unified format — the same field the existing MUBUF
`BufferLoadFormatX` row already reads.

### Source 3 — KytyPS5 `MemoryOps.cpp::DecodeSmem` (L218-256)

Both families get the identical operand shape, with an explicit comment:

> SMEM encodes SBASE in SGPR pairs. Scalar-buffer loads still use the same
> pair index; their descriptor operand consumes four SGPRs from that base.

`inst.offset = SignExtendU32(word1 & 0x1fffff, 21)` **and** `src1 = soffset` are
set together — two simultaneous independent fields, which is why Raeen needs a
three-operand format rather than a choice between them.

### Source 4 — KytyPS5 `spirvEmitterMemory.cpp` (L212-278)

`EmitBufferByteAddress` shows where `stride` *does* apply, and why it cannot
apply to SMEM:

```cpp
uint32_t index = ConstantU32(state, 0);
if (inst.memory.idxen) { index = LoadAddressSource(); }
uint32_t offset = ConstantU32(state, inst.memory.offset);
if (inst.memory.offen) { offset = EmitAddU32(state, offset, LoadAddressSource()); }
const auto soffset = LoadAddressSource();
return EmitBufferAddressFromParts(state, inst, index, offset, soffset);
```

and inside `EmitBufferAddressFromParts`, the non-swizzled (linear) case is

```cpp
const auto linear_index = EmitBinaryU32(state, OpIMul, address_index, stride);
const auto linear       = EmitAddU32(state, linear_index, offset);
...
return EmitAddU32(state, buffer_offset, soffset);
```

So `index * stride` is gated on `idxen` — a flag SMEM does not have. With no
index the buffer offset reduces to `offset + soffset`, matching source 1.
(`reference/mesa` was **not** usable: the local checkout has only
`src/amd/{addrlib,common,registers}`, hence no instruction tables.)

## What now recompiles

### `s_buffer_load*` with a V# base

* **New formats** `Sdst{,2,4,8,16}SvSoffsetOffset` (`src[0]` = V# quad,
  `src[1]` = soffset register, `src[2]` = immediate byte offset) — the parser
  refusal that stopped ASTRO is gone, for **all five widths**, on both the
  next-gen SMEM (`0x3d`) and legacy SMRD encodings.
* **New analysis pass** `shader_capture_vsharp_buffer_loads` — decodes the V#
  from live-in user SGPRs, resolves `base48 + soffset + imm`, and snapshots the
  dwords from guest memory into the existing per-PC
  `embedded_constant_loads` representation. This is what unblocks the GTA V
  shape, where **no storage buffer is bound at all**.
* **One shared recompiler** `sbuffer_load_dwords` replaces five near-duplicate
  bodies. Lowering order: per-PC capture → bound storage buffer → named
  refusal. All five widths now accept a *runtime* offset (previously only `x4`
  did) and the combined `soffset + immediate` form, summed at runtime in the
  uint domain.

Widths covered: 1, 2, 4, 8, 16 dwords, with a NULL soffset (immediate only), a
register soffset (immediate 0), or both.

### MUBUF `BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen]`

New row, composed from two rows that already existed rather than a new rule:

* addressing / `stride` / `buffer_index` from the `BufferLoadFormatX` idxen row
  (`idxen`-only ⇒ `vindex * stride + inst_offset`, no `offen` voffset term);
* the four-channel unpack from
  `Recompile_TBufferLoadFormatXyzw_Vdata4VaddrSvSoffsIdxenFloat4`;
* the format from the descriptor — `(V#.dword3 >> 12) & 0x7f` — which is the one
  real difference between the MUBUF and MTBUF forms (MTBUF carries `dfmt`/`nfmt`
  in the instruction; the `Float4` suffix hardcodes 119).

MUBUF `idxen`/`offen` combinations already wired stay as they were; only the
`Vdata4VaddrSvSoffsIdxen` × `BufferLoadFormatXyzw` cell was missing.

## What still refuses, and why

| Form | Refusal |
|------|---------|
| `s_buffer_load*` where the V# has no bound descriptor **and** the capture cannot resolve | Names type, format, width, `s[base:base+3]`, soffset register and immediate |
| `s_buffer_load*` whose read exceeds the descriptor's own `stride * num_records` bound | Not captured (a `debug` line records the bound); falls through to the refusal above |
| `s_buffer_load*` whose V# quad is written in-shader before the load | Not captured — the user-data snapshot is not what the load will read |
| `s_buffer_load*` with an unprovable register soffset (computed, lane-dependent, `m0`) | Not captured; `resolve_scalar_soffset_bytes` proves only an unwritten live-in user-data register or a preceding `s_mov_b32`/`s_movk_i32` constant |
| `BufferLoadFormatXyzw` whose V# unified format is not 119 | Kyty's `tbuffer_load_format_xyzw` helper serves 32_32_32_32_float and leaves the destination untouched otherwise. A narrower result than a correct fetch — but not a guess, and it replaces a refusal that failed the whole shader. |

## Two correctness guards worth naming

1. **A bound descriptor always wins.** The capture bakes a translate-time
   snapshot into the module; a bound storage buffer reads guest memory live at
   draw time. If the capture shadowed a bound V#, a per-draw constant buffer
   would freeze at frame 1. `shader_capture_vsharp_buffer_loads` therefore skips
   any V# quad already covered by `bind.storage_buffers.start_register`
   (with the same NGG `shift` convention as
   `shader_measure_constant_buffer_accesses_shifted`).

2. **`%buf` gating.** The `sbuffer_load_dword*` SPIR-V helpers index `%buf`,
   which only exists when a storage buffer is bound. Their emission was
   ungated — previously unreachable, because with `buffers_num == 0` translation
   failed before assembly. Now that a capture can serve the load with no
   descriptor bound, emitting the helper text would fail assembly on an
   undefined `%buf`, so the block is gated on `has_buffers` like the MUBUF
   helpers next to it.

3. **Combined immediates as uint constants.** The runtime add happens in the
   uint domain, but `Spirv::add_constant` files an `IntegerInlineConstant` —
   which is what the SMEM parser produces for the sign-extended imm21 — as
   `Int` only. `find_constants` now registers combined-form immediates as uint
   as well; without it any immediate outside the seeded `0..=32` range resolved
   to `unknown_uint_constant` and assembly failed.

## Wiring

`shader_capture_vsharp_buffer_loads` is chained from
`shader_capture_runtime_scalar_loads_shifted`, so the VS and PS stages get it
through their existing call sites. The **compute** stage never called that entry
point, so `raeen-gpu`'s `translate_cs` calls the new pass directly — ASTRO's
measured blocker is in three *compute* shaders, so without that call site the
parser fix alone would have changed nothing for it.

## Tests

10 new tests, failing-first verified.

* parse (encode-bytes → decoded instruction): combined V# form decodes three
  operands; all five widths; a zero immediate keeps the two-operand shape.
* analysis: a read past `num_records` is not captured; a bound descriptor is not
  shadowed.
* recompile, spirv-val-validated (Vulkan 1.3): GTA V's `x8` shape with **no
  storage buffer bound**; the combined form through the capture; the combined
  form through a bound descriptor (with a runtime `OpIAdd` and a >32 immediate);
  all five widths through a bound descriptor; Avatar's
  `BufferLoadFormatXyzw` idxen row.

spirv-val is the gate throughout — naga has documented false negatives in this
crate (it rejects `OpImageGather` and storage-image-write modules outright).

Counts: kyty-graphics **532 lib + 6 integration** (from 524 + 6), raeen-gpu
**308 lib** + all integration suites, raeen-hle **567**.
`dispatch_table_counts` 358 → **364** rows, 350 → **356** implemented, `ni`
unchanged at 8.

## Not claimed

No title was re-run. This is a decode / translate / wiring and test claim, not a
rendering claim for GTA V, ASTRO.BOT or Avatar. In particular the
`BufferLoadFormatXyzw` row unblocks the shader without guaranteeing a correct
fetch for non-119 descriptor formats, and the capture path is a translate-time
snapshot whose freshness is only guaranteed for buffers the title does not
rewrite between translation and the draw.
