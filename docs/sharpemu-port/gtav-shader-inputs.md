# GTA V vertex-input shader blocker — `registers_num/input format: 2/5`

**Date:** 2026-07-28
**Branch:** `port-gtav-shader`
**Crates touched:** `kyty-graphics`, `raeen-gpu`
**Measured against:** the live 9-title baseline on build `2e4bdca` — GTA V at
stage `timed_out` with **192 presented flips**, zero unresolved NIDs, no canary
smash. Its only remaining first-order blocker was shader translation:

```
ERROR kyty_graphics::shader::spirv: Spirv::WriteGlobalVariables:
  not supported: invalid registers_num/input format: 2/5
```

This is a translation-correctness fix with in-tree tests. **It is not a claim
that GTA V now renders** — the title has not been re-run for this change.

---

## 1. What `2/5` actually is

The refusal formatted `{registers_num}/{V# unified format}`, so `2/5` decodes as
two independent quantities that the old code conflated into one hand-picked
match:

| Field | Value | Meaning |
|-------|-------|---------|
| `registers_num` | `2` | `ShaderSemantic::size_in_elements()` — how many VGPRs the vertex fetch writes, i.e. how many components the shader consumes. Set in `shader_parse_attrib` (`analysis.rs`) from the semantics table, **not** from the descriptor. |
| unified format | `5` | The V#'s 7-bit RDNA2 unified FORMAT field (`ShaderBufferResource::format()` = dword 3 bits 12..18). Unified **5 = (dataFormat 1, numberFormat 4) = `8_UINT`** — one 8-bit **raw unsigned integer** component. |

Format 5's decode is confirmed three ways:

* `SharpEmu.ShaderCompiler/Gfx10UnifiedFormat.cs` (RDNA2 ISA table 47):
  `5 => (1u, 4u)` — dfmt 1 = `8`, nfmt 4 = `UINT`.
* This crate's own numeric classifier `SampledClass::from_unified_format`
  already lists `5` in its **Uint** arm.
* This crate's own host mapping `gen5_vertex_format_and_size`
  (`raeen-gpu/src/draw_translate.rs`) already maps `5 => (R8_UINT, 1 byte)`,
  with a comment recording it as measured on *"GTA V's first submitted DCB"*.

So the pair is **two raw-integer components**. The component count exceeding the
hardware format's single channel is *normal*: GCN vertex fetch fills channels
beyond the descriptor with the `(0, 0, 0, 1)` default, exactly as Vulkan fills
components a narrower `VkFormat` does not supply. The crate already documents
and handles that width mismatch in the other direction.

## 2. Why it was rejected

Three separate sites each carried their own `match` over the same
`(registers_num, is-raw-integer)` pair, and each covered a **different subset**:

| Site | Accepted pairs |
|------|----------------|
| `Spirv::WriteGlobalVariables` (`spirv.rs`) — the `%attrN` `OpVariable` | `(1,uint)`, `(1..=4, float)` |
| `Recompile_Fetch` (`recompile.rs`) — the `OpLoad` + bitcast + fetch helper | `(1,uint)`, `(1..=4, float)` |
| `RAEEN_VS_PASSTHROUGH` diagnostic (`recompile.rs`) | `(1,uint)`, `(1..=4, float)` |

Raw-integer support had been added for **one component only** — Minecraft's
format-11 (`16_UINT`) bone index. Every other width fell through to the
catch-all `Err`, which refuses the **entire vertex shader** and therefore every
draw that uses it.

A float declaration would not have been a valid fallback either: Vulkan requires
the shader input's numeric class to match the bound attribute's `VkFormat`
class, and the host already binds `R8_UINT` for format 5. The guest also
consumes these VGPRs with integer bit operations, so a numeric float convert
would select the wrong value even where validation permitted it.

## 3. What was implemented

**One shared resolver** replaces the three divergent matches:

* `kyty-graphics/src/shader/spirv.rs`
  * `SampledClass::scalar_type_str()` — `float` / `uint` / `int`.
  * `vertex_input_class(&ShaderBufferResource) -> SampledClass`.
  * `vertex_input_types(registers_num, class) -> Option<VertexInputTypes>` —
    the pointer type for the declaration, the load type, the same-width float
    type (bitcast target and fetch-helper parameter type), the float scratch,
    the `fetch_*` helper, and whether a bitcast is required. Supports the full
    **4 widths x 3 numeric classes = 12 pairs**.
  * `write_types` gained `%_ptr_Input_v2uint`, `%_ptr_Input_v4uint`,
    `%_ptr_Input_v{2,3,4}int` (`v3uint` and scalar `int`/`uint` already existed).
  * `vertex_input_pair_skips()` — process-wide counter of refused pairs.
* `kyty-graphics/src/shader/recompile.rs`
  * `recompile_fetch` and `vs_passthrough_source` now consume the resolver.
    The bitcast target became width-parameterised (it was hardcoded `%float`,
    which is invalid for a vector load).
* `raeen-gpu/src/agc_exec.rs`
  * `record_shader_skips` reports `vertex_input_pair_skips` alongside
    `texture_cap_skips` / `storage_addressing_skips`.

Refusals now name the exact pair instead of a bare number:

```
vertex attribute 0: 7 components of unified format 5 (Uint)
  — only 1..=4 components are supported
```

**GTA V's `2/5` case now translates**: `%attr0 = OpVariable %_ptr_Input_v2uint
Input`, `OpLoad %v2uint`, `OpBitcast %v2float`, `OpStore %temp_v2float`,
`OpFunctionCall %void %fetch_f1_f1_vf2_ %v2 %v3 %temp_v2float`, assembled and
**spirv-val clean**.

## 4. Reference basis

`Gen5SpirvTranslator.DeclareVertexInputs`
(`reference/sharpemu/src/SharpEmu.ShaderCompiler.Vulkan/Gen5SpirvTranslator.cs`
L1307-1353, read from the live working tree — upstream's `6db095e` revert and
`db4339f` restore make per-commit reads misleading) builds the interface type as
`componentKind(numberFormat) x componentCount` for **all** of 1..=4 components
and all three numeric classes, rather than enumerating a subset:

* `numberFormat 4 => Uint`, `5 => Sint`, everything else `Float`
* `1 => componentType`, `2..=4 => TypeVector(componentType, count)`
* `ComponentCount` comes from the fetch instruction's dword count
  (`Gen5ShaderScalarEvaluator.TryCreateVertexInputBinding`), and the numeric
  class from the buffer descriptor — the same two-independent-quantities model.
* `TryEmitVertexInputFetch` (L3234) bitcasts non-uint components rather than
  converting them.

Kyty upstream (`reference/kyty/source/emulator/src/Graphics/ShaderSpirv.cpp`
L7229) declares **float only** and `EXIT`s on any other width — it has no
integer vertex-input concept at all, so the whole raw-integer axis is beyond it.

Mesa was consulted for the unified-format table's hardware meaning; nothing was
copied.

## 5. Tests

| Test | Location | Gate |
|------|----------|------|
| `gta_two_channel_uint_vertex_attribute_translates_to_validated_spirv` | `crates/kyty-graphics/tests/gcn_to_spirv.rs` | Public API only: `shader_parse_attrib` (real analysis derives `registers_num = 2` from the semantic and pairs it with a format-5 V#) → `shader_recompile_vs` → **spirv-val** (Vulkan 1.3) + naga entry-point check |
| `two_channel_uint_vertex_fetch_loads_a_uint_vector` | `kyty-graphics/src/shader/recompile.rs` | Declaration, load width, componentwise bitcast, scratch store, and both VGPR channels reaching the helper |
| `every_supported_vertex_input_pair_names_declared_types` | same | All 12 pairs name a pointer type the SPIR-V prelude actually declares, and only the integer classes are reinterpreted |
| `unsupported_vertex_input_pair_is_named_and_counted` | same | Out-of-range widths are refused, named with the pair, and counted in `vertex_input_pair_skips` |
| `parse_attrib_carries_the_measured_gta_two_component_uint_pair` | `kyty-graphics/src/shader/analysis.rs` | The `(2, format 5)` pair is what the semantics table + V# actually produce |

The failing-first property was verified directly: with the pre-fix pair coverage
restored, the integration test fails at the same site
(`Spirv::WriteGlobalVariables`).

Counts after the change: **kyty-graphics 498 lib + 6 integration**
(was 494 + 5), **raeen-gpu 294 lib + all integration suites** green.
`cargo clippy -p kyty-graphics -p raeen-gpu --all-targets -- -D warnings` clean.

`spirv-val` is the gate for this module, not naga: naga cannot validate this
crate's storage-image-write and descriptor-array shapes in general (a documented
false negative — `InvalidArrayBaseType`).

## 6. Open / not addressed

* **Host `VkFormat` coverage for SINT vertex buffers.** The SPIR-V side now
  declares `int`-typed inputs for the `Sint` class (unified 6, 12, 19, 21, 28,
  …), but `gen5_vertex_format_and_size` maps no SINT row yet, so such a V#
  still fails on the host with a named error before reaching the shader. The two
  sides now *agree*; adding the rows is a separate, mechanical follow-up.
* **Whether `2/5` was GTA V's only shader blocker.** The baseline named it as
  the first one. A re-run is required to find what is next; no rendering or
  milestone claim is made here.
