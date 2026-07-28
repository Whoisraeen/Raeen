# SharpEmu Gen5 shader / AGC port — findings

Batch date: 2026-07-28. Branch `worktree-agent-a7b94c269712ce35c`.

Source (read-only, gitignored): `reference/sharpemu`, working tree at
`refs/heads/main`. Ports cite the originating SharpEmu `file:line` in doc
comments.

Motivating measured failure (ASTRO.BOT):

```
kyty_graphics::shader::spirv: storage_texture_dim_format: not supported:
mixed storage image dims/formats in one shader ((Three, "Rgba16f") vs (Two, Rgba16f))
```

followed by a host crash.

---

## Headline result

**The mixed 2D + 3D storage-image case now translates.** The per-binding
storage model that replaces the shader-wide refusal had already landed in
`kyty-graphics` before this batch; what was missing was an acceptance test
proving it, and the *host* half of the same contract. Both are now in tree.

New acceptance test — `crates/kyty-graphics/src/shader/recompile.rs`
`astro_mixed_2d_and_3d_rgba16f_storage_images_translate_to_two_image_types`.
One compute shader writes a 3D `Rgba16f` volume and a 2D `Rgba16f` target; the
module must declare two distinct `OpTypeImage` types, bind each at its own
descriptor slot, index each from the body that writes it, build a `v3uint`
coordinate for the 3D write, and pass real **spirv-val** (Khronos, Vulkan 1.3).

> Validation note: `naga` cannot gate this path. Its SPIR-V front end rejects
> *every* `OpImageWrite` storage module this generator produces with
> `InvalidImage` — a homogeneous 2D-only module fails identically (measured
> during this batch with a throwaway probe over 2D-only / 3D-only / mixed).
> That is a known naga false negative, the sibling of the existing
> `InvalidArrayBaseType` carve-out, so the test gates on spirv-val instead.

---

## Per-commit verdicts

### Priority 1

| Commit | Subject | Verdict |
|---|---|---|
| `5228335` (#587) | support Gen5 flat memory and 3D images | **ALREADY-HAVE (SPIR-V half) + PORTED (host half)** |

The translator side was already per-binding, not per-shader:
`StorageFormat`, `storage_key_of`, `storage_key_ordinal`, `storage_key_suffix`,
`storage_keys_present`, `storage_key_layout` in
`crates/kyty-graphics/src/shader/spirv.rs`, consumed by
`shader_calc_binding_indices` (`analysis.rs`) and routed per site by
`storage_site_route` / `route_storage_ids` (`recompile.rs`). MIMG DIM already
decodes per instruction via `SampledDim::from_texture_type`, and the 3D write
already builds a three-component integer coordinate. Depth already transports
from the T# DEPTH field through `TextureUpload::depth` /
`StorageImageUpload::depth` into `extent.depth`, and the byte-count and
staging arithmetic already multiply by depth.

**The real gap, now fixed:** the host derived "is this a volume?" from the
*slice count* (`depth > 1`) while the recompiler derives `Dim3D` from the T#
**TYPE nibble** alone. A type-10 descriptor whose DEPTH field is 0 is a legal
one-slice volume with `depth == 1`, so the host built a `VK_IMAGE_TYPE_2D`
image and a `TYPE_2D` view underneath a `Dim3D` image type. That is the same
emit/bind divergence class that already cost a device loss for the arrayed
case, which is precisely why `TextureUpload::array` is type-driven. Measured
shape: GTA V's tile-5 single-voxel type-10 T#
(`draw_translate.rs::gta_tile5_single_voxel_volume_reads_the_origin_texel`
asserts it decodes to `(1, 1, 1)`).

Fix: a type-driven `volume: bool` on both `TextureUpload` and
`StorageImageUpload` (`crates/raeen-gpu/src/vulkan/offscreen.rs`), set in
`draw_translate::decode_texture` / `read_storage_image`, consumed at all four
create/view sites (`offscreen.rs`, `vulkan/compute.rs`), and added to both
cache keys (`vulkan/cache.rs` `TextureKey` / `ComputeImageKey`) so a one-slice
3D image can never be served under a plain-2D descriptor sharing the same
base/extent/format.

Test: `crates/raeen-gpu/src/draw_translate.rs`
`one_slice_type10_volume_stays_a_3d_image_not_a_2d_one` — pins both the sampled
and the storage side, and pins that a type-9 T# is still not a volume.

Residual, unchanged by this batch: only tile mode 0 (linear) is implemented for
volumes, plus a narrow 1x1x1 tile-5 case; other tiled volumes are a named
refusal. The 3D `vkCreateImage`/`vkCreateImageView` path still has no
device-level test (every `TextureUpload` literal in
`crates/raeen-gpu/tests/` sets `depth: 1`).

### Priority 2

| Commit | Subject | Verdict |
|---|---|---|
| `3574a3b` (#466) | lower VOP3P `V_FMA_MIX_F32/LO/HI` | **PORTED** |
| `472fc96` (#460) | clamp modifier on packed f16 VOP3P ops | **PORTED** |
| `3005bab` (#420) | `v_pk_fma_f16` with exact single rounding | **PORTED with a named deviation** |
| `20eda44` (#465) | wave mask consumed as a per-lane predicate at the lane bit | **DEFERRED — scoped below** |

Raeen had **no VOP3P support whatsoever**. GFX10 moved VOP3P to its own
`0b110011000` prefix (`instruction >> 26 == 0x33`) and `shader_parse` had no
`0x33` arm, so every packed word fell into the catch-all and returned
`ShaderParseError::UnknownEncoding` — one packed instruction dropped the whole
shader. That is exactly SharpEmu's "was dropping Unity HDR shaders".

Ported:

- `crates/kyty-graphics/src/shader/parse.rs` `shader_parse_vop3p` — from
  SharpEmu `Gen5ShaderTranslator.cs` `DecodeVop3p` (opcode table) and its
  `Gen5ShaderEncoding.Vop3p` operand/control case (field layout, including
  `op_sel_hi` split across both dwords: bits [1:0] in word1 [28:27], bit [2] in
  word0 [14]). Opcodes 0x0e/0x0f/0x10/0x11/0x12 (`v_pk_fma/add/mul/min/max_f16`)
  and 0x20/0x21/0x22 (`v_fma_mix_f32` / `_mixlo_f16` / `_mixhi_f16`). An
  unported packed opcode (the integer ops, dot2/dot4) is a **named refusal**
  after the instruction length is computed correctly, so the failure is
  attributable to its own pc instead of desynchronizing the rest of the stream.
- `crates/kyty-graphics/src/shader/types.rs` `Vop3pControl` — from SharpEmu's
  `Gen5Vop3pControl`. These modifiers cannot live on `ShaderOperand` the way
  VOP3A's `negate`/`absolute` do: VOP3P carries two negate masks (the two
  16-bit lanes negate independently) and `op_sel`/`op_sel_hi` select a half per
  lane; one `bool` per operand cannot express that.
- `crates/kyty-graphics/src/shader/recompile.rs`
  `recompile_vop3p_packed_f16` / `recompile_vop3p_fma_mix` plus helpers
  (`vop3p_clamp`, `vop3p_half_operand`, `vop3p_min_max`, `vop3p_store`) — from
  `Gen5SpirvTranslator.Alu.cs` `TryEmitPackedF16` / `TryEmitFmaMix` /
  `EmitPackedF16Operand` / `EmitPackedF16MinMax` /
  `EmitClampToUnitInterval` / `EmitFmaMixOperand`. Eight new dispatch rows.

Semantics preserved: each lane widens its selected source half to f32,
negates per `neg_lo`/`neg_hi`, runs the op in f32, and narrows back to f16.
`clamp` saturates to `[0, 1]` with **ordered** compares so NaN becomes 0
without a separate `IsNan` test — note the pre-existing VOP3 bodies *refuse*
clamp by name, so a packed op had to implement it. `min`/`max` are
`fminnum_like`/`fmaxnum_like` (a NaN operand yields the other). For the mix ops
the same fields are reinterpreted: `op_sel_hi` bit *i* means "read `src[i]` as
an f16", `neg_hi` is the absolute-value modifier and `neg_lo` negates,
abs-then-neg; `_MIXLO`/`_MIXHI` merge the narrowed result into one half of
`vdst` and leave the other intact.

**Named deviations from SharpEmu, both deliberate:**

1. f16↔f32 conversion uses GLSL `UnpackHalf2x16` / `PackHalf2x16`, matching
   this crate's existing `VCvtF32F16` / `VCvtPkrtzF16F32` bodies, rather than
   SharpEmu's explicit branchless integer sequences. Those exist to pin
   subnormal/rounding behaviour without float-controls execution modes; the
   GLSL ops leave that to the driver.
2. `v_pk_fma_f16` lowers to a single f32 `Fma` then the f16 narrowing, **not**
   to SharpEmu's round-to-odd 2Sum sequence (#420 "exact single rounding"). The
   2Sum residual is error-free only if every op in the chain carries
   `OpDecoration NoContraction`, and this generator emits decorations in a
   separate `write_annotations` phase with **no per-body injection point** — an
   uncorrected 2Sum measurably decays to the double-rounded answer anyway
   (SharpEmu observed exactly that on RDNA3 Windows). So the result can differ
   from hardware in the last f16 bit on midpoint inputs. Getting the shader to
   translate at all is the win: it was previously dropped whole. Closing this
   properly needs a decoration hook in `Spirv`, then the 12-line round-to-odd
   sequence per lane.

Tests (all in `recompile.rs`):
`vop3p_encoding_no_longer_drops_the_whole_shader` (decode + the named refusal
for an unported packed opcode), `vop3p_packed_and_mix_ops_lower_to_valid_spirv`
(all eight opcodes, clamp and negate set, spirv-val clean),
`vop3p_lane_shapes_are_packed_not_scalar` (two lanes extract different halves
and repack into one dword; the mix ops are a single fma that merges into one
half and preserves the other; `op_sel_hi` picks f16-vs-f32 per source).

Collateral: `shader::parse::tests::unknown_encoding_is_typed_error` used
`instr >> 26 == 0x33` precisely *because* it matched no family. It now uses
`0x39` (0x38 is MUBUF, 0x3a MTBUF).

#### `20eda44` (#465) — deferred, with the scope measured

SharpEmu splits two mechanisms that are easy to conflate:

- **(a) the lane-bit predicate** — `GuestWaveLane`, `CurrentLaneBit`,
  `IsCurrentLaneSet`/`IsWaveMaskActive`, `BooleanToLaneMask`, `StoreWaveMask`
  (`Gen5SpirvTranslator.cs:5231-5289`, `5405-5429`) plus `LoadS64`/`StoreS64`/
  `GetRawSource64` (`Alu.cs:3065-3130`). ~140 lines, purely local to one
  invocation — a shift, an AND and a compare. For **all graphics stages**
  SharpEmu itself runs this in the degenerate lane-0 mode
  (`UsesSubgroupOperations()` is compute-only), so `CurrentLaneBit()` is
  literally `1`.
- **(b) the wave-wide producer** — `BooleanToWaveMask` emitting
  `OpGroupNonUniformBallot`, the wave64 emulation with workgroup barriers and
  LDS scratch, `SubgroupAny`. ~250 lines, compute-only, and a structural
  rewrite here.

Only (a) is in scope, and its payoff is **not** per-lane semantics — it is
correctness for *complemented-mask* idioms. `IsNotZero64(mask)` reports
"active" when `S_NOT_B64`/`S_ORN2_B64` sets the unused upper 63 bits;
`IsNotZero64(mask & lane_bit)` does not.

Raeen's position today, precisely:

- **`v_cndmask_b32` is already correct.** `recompile_vcndmask_b32` masks with
  `%uint_1` before the compare (`recompile.rs`, the
  `OpBitwiseAnd %uint %t22_<index> %uint_1` line) — identical to
  `IsCurrentLaneSet` with `CurrentLaneBit() == 1`.
- **Store/exec predication is the residual bug.** All 22 exec-predicated body
  tails test `OpINotEqual %bool %exec_lo_u_<index> %uint_0` — the **whole
  word**. After `s_not_b64 exec, exec`, `exec_lo` is `0xFFFFFFFE`: the test says
  active while lane 0's bit is clear. The fix is to AND with `%uint_1` first, at
  each of those 22 inline sites.

Not attempted in this batch: 22 hand-written sites, each with golden-text
assertions in existing tests, is a wide mechanical sweep that needs its own red
test per site. Doing it half-way is worse than naming it. Note also that even
in SharpEmu, `s_cbranch_vccnz`/`execnz` do **not** use the lane bit (they use
`SubgroupAny` on a bool), so this is not a branch-correctness fix.

### Priority 3

| Commit | Subject | Verdict |
|---|---|---|
| `5f97031` (#514) | larger bounded Gen5 programs | **N/A** — no program-size cap exists in `kyty-graphics`' parse/analysis path to raise |
| `8e1e89c` (#545) | Gen5 hull shaders omitting PGM_LO/HI in CreateShader | **ALREADY-HAVE** — `crates/raeen-hle/src/libsce_agc.rs` searches the program-register table rather than assuming entries 0/1, with the GTA V hull-header rationale in the comment at the `SPI_SHADER_PGM_LO_HS`/`_LS` table |
| `f9d9213` (#556) | merge Prospero attrib-table formats onto IR vertex inputs | **ALREADY-HAVE** — the `vertex_attributes` pipeline in `crates/raeen-gpu/src/draw_translate.rs` |
| `8e5a0bf` (#558) | `sceAgcDcbSetUcRegisterDirect` | **ALREADY-HAVE** — registered in `libsce_agc.rs`, with the `*GetSize` sibling |
| `74a5198` (#535) | missing Cb/Dcb GetSize stubs | **ALREADY-HAVE** — the `*GetSize` family in `libsce_agc.rs` |
| `a709ccc` (#395) | guest image byte-count calculation | **ALREADY-HAVE** — `expected_sampled_bytes` / `expected_storage_image_bytes` in `draw_translate.rs`, both depth- and layer-aware and unit-tested |
| `2bb2ad3`-era (#553) | MRT / fast-clear | **ALREADY-HAVE** — landed in Raeen before this batch |

Revert trap (`6db095e` / `db4339f` #650) respected: every verdict above was
formed by reading the reference **working tree at its current tip**, never a
lone historical commit.

---

## Test counts

| Suite | Before | After |
|---|---|---|
| `kyty-graphics` lib | 490 (489 pass, 1 pre-existing fail) | **494 pass** |
| `kyty-graphics` integration | 5 | 5 |
| `raeen-gpu` lib | 284 | **285 pass** |
| `raeen-gpu` integration | 45 across 18 suites | 45, all green (1 ignored, unchanged) |

`shader::recompile::tests::dispatch_table_counts` was **already red on the
branch tip** before this batch (340 rows against an expected 337 — three rows
had been added without updating the bookkeeping assertion). It is now correct
at 348/340 including the eight VOP3P rows.

## Follow-ups worth their own change

1. **Exec-mask lane bit** (#465, above): AND `exec_lo` with `%uint_1` at the 22
   exec-predicated body tails in `recompile.rs`.
2. **Decoration hook in `Spirv`**, then `v_pk_fma_f16`'s round-to-odd 2Sum
   (#420) for exact single rounding.
3. **Device-level 3D image test** in `crates/raeen-gpu/tests/` — no existing
   `TextureUpload`/`StorageImageUpload` literal there sets `depth > 1`, so the
   `vkCreateImage`/`vkCreateImageView` 3D path is unit-tested only.
4. **General 3D tile-mode detile** — volumes outside tile mode 0 (plus the
   1x1x1 tile-5 special case) remain a named refusal.
