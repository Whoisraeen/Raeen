# SMEM scalar loads: register soffset, and the vertex stage that never resolved anything

**Date:** 2026-07-28
**Baseline measured:** build `2741d21`, `artifacts/compat/post-shaderfix-validated-20260728.json`
**Branch:** `port-smem-soffset`
**Scope:** `crates/kyty-graphics/src/shader/{types,parse,analysis,spirv,recompile,resources}.rs`,
`crates/raeen-gpu/src/shader_fetch.rs`

---

## 1. What the measurement actually said

Three titles named a scalar-load blocker. They are **not** the same defect.

| Title | Stage / flips | First blocker | Real cause |
|-------|---------------|---------------|------------|
| Avatar: Frontiers of Pandora | `timed_out`, 256 flips, 1597 shader errors | `can't recompile: SLoadDwordx16 [Sdst16SbaseSoffset] s[12:27], s[8:9], 0` | Vertex stage never ran the runtime scalar-load capture |
| Grand Theft Auto V | `timed_out`, 256 flips, 4 shader errors | `can't recompile: SLoadDwordx4 [Sdst4SbaseSoffset] s[20:23], s[12:13], 64` | Same |
| ASTRO.BOT | `rendering`, 36 flips, 45 shader errors | `parse: not implemented smem feature: offset != 0 with register soffset` | Genuinely the register soffset |

Two pieces of evidence decide (A) against the register-soffset hypothesis:

1. **The printed third operand is a constant, not a register.**
   `types.rs::operand_to_str` renders `IntegerInlineConstant` as a bare decimal
   and `Sgpr` as `sN`. Avatar prints `0` and GTA V prints `64`; a register
   soffset would have printed `s4`. So both are ordinary immediate-offset loads.
2. **The message class is `CannotRecompile`, not `UnknownTypeFormat`.**
   `spirv.rs` distinguishes `can't recompile (no table entry for …)` from bare
   `can't recompile: …`. The bare form means a dispatch row *existed* and its
   function returned `Ok(false)`. The x16 width and the `Sdst{N}SbaseSoffset`
   formats were already wired by the previous pass; nothing was missing there.

And every occurrence of both messages in both logs is a **vertex** shader
(`OpEntryPoint Vertex %main`, and `raeen_gpu::shader_fetch` reports
`stage="vs"`).

## 2. Root cause A — `translate_vs` never ran the capture pass

`sload_dword_extended` returns `Ok(false)` when there is no per-PC capture for
the load *and* `bind.extended.used` is false — i.e. the shader has no EUD
window, so the only way a non-EUD SGPR-pair base can be resolved is the
analysis pass `shader_capture_runtime_scalar_loads`, which reads the pointer out
of the live user-data file and snapshots the target dwords.

That pass was called from `translate_ps` and `translate_cs` in
`crates/raeen-gpu/src/shader_fetch.rs` — and **not** from `translate_vs`. The
pixel call site's comment even asserted the opposite:

> `// PC-relative scalar constant tables are stage-agnostic. VS and CS already run this capture;`

They did not. VS ran only `shader_detect_embedded_constant_loads` (PC-relative
bases) and `shader_detect_embedded_buffer_fetch`.

### The rebase that made this non-trivial

A next-gen vertex program is required to be a gs-prolog
(`shader_get_input_info_vs_decoded` errors with `vs: next-gen without
gs-instead-of-vs` otherwise) and therefore addresses user data **eight SGPRs
up**: shader register `N` is hardware user-data slot `N - 8`. That constant
already exists three times in-tree:

* `analysis.rs::rebase_ngg_constant_sharps` — `const NGG_SCALAR_BASE: usize = 8`,
  `hardware_slot = scalar_reg - NGG_SCALAR_BASE`
* `shader_measure_constant_buffer_accesses_shifted(&code, bind, if gs_prolog {8} else {0})`
* the recompiler's `shift_regs` in `sload_dword_wide` / `recompile_sload_dword`

So the fix is `shader_capture_runtime_scalar_loads_shifted(code, mem, user_sgpr,
shift, bind)` (the old entry point delegates with `shift = 0`), called from
`translate_vs` with `shift = 8` when `vs_info.gs_prolog`, and with the
gs-instead-of-vs user-SGPR file (`vs.gs_user_sgpr`) rather than `vs.vs_user_sgpr`
— mirroring `shader_get_input_info_vs_decoded`'s own selection.

Avatar's `s[8:9]` is hardware slot 0:1; GTA V's `s[12:13]` slot 4:5. The
unshifted entry point resolves **neither** (test:
`vertex_stage_shift_maps_shader_registers_to_hardware_user_data_slots` asserts
both halves).

## 3. Root cause B — the RDNA2 combined addressing mode (ASTRO.BOT)

### The rule, with sources

```
address     = base_pair_u64  +  zext64(SGPR[soffset])  +  sext21(imm_offset)
address    &= !3
dword_index = address >> 2
```

* both offset terms are in **bytes**
* the immediate is **21-bit signed** (`word1 & 0x1fffff`, sign-extended)
* the soffset register value is **unsigned 32-bit**
* they **add** — neither replaces the other
* the sum is dword-aligned by **truncation**, not by faulting

Established from two independent references (`reference/mesa` has only
`src/amd/{addrlib,common,registers}` locally — no compiler tree, so no SMEM
table; it was not usable here):

| Reference | File / lines | What it shows |
|-----------|--------------|---------------|
| KytyPS5 (MIT) | `src/graphics/shader/recompiler/MemoryOps.cpp` `DecodeSmem` (~L218-256) | `inst.offset = SignExtendU32(word1 & 0x1fffff, 21)` **and** `DecodeScalarSource(soffset, …, inst.src1)` — two simultaneous independent fields |
| KytyPS5 | `spirvEmitter/spirvEmitterMemory.cpp` `EmitRelativeAddress` (L286-315), called from `EmitSLoadDword` (L1174) with `align_components = true` | immediate `& ~3`, each added source `& ~3`, 64-bit add with carry, then `index = address >> 2` |
| SharpEmu (GPL-2.0) | `src/SharpEmu.ShaderCompiler/Gen5ShaderTranslator.cs`, `Gen5ShaderEncoding.Smem` case | `Gen5ScalarMemoryControl(count, offset, dynamicOffsetRegister)` — again both fields at once |
| SharpEmu | `src/SharpEmu.ShaderCompiler/Gen5ShaderScalarEvaluator.cs` L1889-1900 | `byteOffset = immediateOffset + dynamicOffset;` `address = (baseAddress + byteOffset) & ~3UL;` |

Read from the live working trees, not historical commits (upstream revert
`6db095e` was later restored by `db4339f`).

### Representation

Because both terms are simultaneous, `src[1]` alone cannot carry the offset. The
parser previously *collapsed* a NULL soffset into `src[1]` as the immediate
(Kyty's shape, and a lot of code depends on it), so that shape is untouched.
When **both** terms are live the instruction takes a three-operand format:

```
src[0] = SGPR base pair
src[1] = soffset register
src[2] = immediate byte offset       (IntegerInlineConstant, sign-extended imm21)
```

Five new formats — `SdstSbaseSoffsetOffset`, `Sdst2SbaseSoffsetOffset`,
`Sdst4SbaseSoffsetOffset`, `Sdst8SbaseSoffsetOffset`,
`Sdst16SbaseSoffsetOffset` (`format_define(&[DA{N}, S0A2, S1, S2])`,
following the existing `FlatAddr = [D, S0A2, S1A2, S2]` precedent for an
immediate in `src[2]`).

A register soffset with a **zero** immediate keeps its old two-operand format —
minimum blast radius, so nothing that already parsed changes shape.

Two shared accessors in `types.rs` keep consumers honest:

* `smem_offset_operand(inst)` — `src[2]` for the three-operand forms, else
  `src[1]`. Every place that wants "the constant part of the address" goes
  through this, so a three-operand load is never silently read as offset 0.
* `smem_register_soffset(inst)` — `Some(operand)` when the soffset is a runtime
  register (covers both shapes).

### `s_buffer_load` still refuses, by name

Opcodes `0x08`-`0x0c` address through a four-SGPR V#, so the pointer rule above
does not transfer and nothing measured needs it. The parser refuses with
`offset != 0 with register soffset on an s_buffer_load (V# base)`.

### Resolving the soffset

A runtime soffset makes the address unknowable at translate time *in general*.
`analysis.rs::resolve_scalar_soffset_bytes` proves it in exactly two bounded,
side-effect-free shapes:

1. the register is **never written** before the load and names a captured live-in
   user-data slot (`user_sgpr.value[reg - shift]`) — the SRT/global-table
   pointer ABI;
2. the **last writer** before the load is `s_mov_b32` / `s_movk_i32` from a
   compile-time constant.

Anything else — a computed offset, `m0`, `vcc` — returns `None`, the load is not
captured, and the recompiler keeps a named refusal. A resolved load folds its
whole address (`base + soffset + imm`, masked) into the **existing** per-PC
`embedded_constant_loads` capture, which `sload_dword_extended` already
materializes as SPIR-V constants — so no new lowering path was needed.

The same resolver runs in `shader_detect_embedded_constant_loads` (PC-relative
bases), where there is no user-data file, so only shape 2 can apply.

### `s_load_dword` (x1)

`recompile_sload_dword` was an embedded-fetch-only gate: any other single-dword
scalar load returned `Ok(false)` → "can't recompile", failing the whole shader.
It now shares `sload_dword_wide(n = 1)`, and x1 joined the width tables in
`shader_capture_runtime_scalar_loads`,
`shader_detect_embedded_constant_loads` and `shader_detect_eud_raw_window`
(where capturing x1 previously "would record dwords nothing can consume" — true
before, false now).

## 4. Honesty: the raw EUD window must not lie

`shader_detect_eud_raw_window` sizes `%eud_raw` as *highest constant dword index
any raw `s_load` addresses, plus one*. It used to `continue` silently past any
load whose offset was not a constant. With register soffsets now parseable that
silence is dangerous: the recompiled raw read clamps against
`OpArrayLength` and yields **0** beyond the window, so an under-sized window
turns a load into silence rather than an error.

The pass now:

* includes x1 in its width table (it was 2/4/8/16);
* uses `smem_offset_operand` rather than `src[1]` directly;
* on a register-soffset load off the EUD base that no per-PC capture covers,
  sets `ShaderEudRawResources::unresolved_dynamic_offset` and `warn!`s with the
  pc, type, format, width and base register — the window becomes an explicit
  **lower bound**, not an authoritative size.

`sload_dword_extended` then refuses the raw-window read by name whenever that
flag is set, instead of reading a window it knows may be short. A shader with no
dynamic-offset load is unaffected (test:
`eud_raw_window_records_and_refuses_an_unresolved_dynamic_offset` asserts both
directions, and the clean case still emits `OpArrayLength %uint %eud_raw 0` and
is spirv-val clean).

## 5. What still refuses, and why

| Form | Refusal |
|------|---------|
| Register soffset whose value cannot be proven (computed, `m0`, `vcc`, or beyond the captured user-data file) | `unresolved register soffset: {type} [{format}] x{N} dwords, base=sB, soffset=sS, imm=0xI, pc=0xP` |
| `s_buffer_load*` with a register soffset **and** a non-zero immediate | `offset != 0 with register soffset on an s_buffer_load (V# base)` — the V# base has different semantics and no reference read here establishes it |
| Raw EUD window read while the window is a known lower bound | `raw EUD window is a lower bound (an s_load off sB has an unresolved register soffset); refusing dword D of xN at pc=0xP` |
| `glc != 0`, `dlc != 0`, literal sbase/soffset | unchanged Kyty refusals |

All of these go through the same `tracing::error!`/`ShaderRecompileError` paths
the compat harness counts as `shader_errors`, so nothing became invisible.

## 6. Tests

14 new (kyty-graphics lib **510 → 524**; integration 6 unchanged):

*Parse, from encoded ISA words* (`parse.rs`)
* `smem_register_soffset_with_immediate_offset_decodes_three_operands` — GTA V's
  exact addressing plus a register soffset; asserts operand slots and both
  shared accessors
* `smem_register_soffset_with_offset_covers_every_width` — opcodes 0x00-0x04
* `smem_register_soffset_without_offset_keeps_two_operand_format` — no regression
* `smem_register_soffset_immediate_sign_extends_21_bits`
* `smem_buffer_load_register_soffset_with_offset_refuses_by_name`

*Analysis* (`analysis.rs`)
* `register_soffset_adds_to_the_immediate_when_it_is_a_live_in_user_sgpr` —
  including a negative control: with the soffset register set to 0 the address
  changes and nothing is captured, proving the term participates
* `register_soffset_resolves_from_a_preceding_constant_move` — the in-shader
  `s_mov_b32` beats a deliberately wrong live-in value
* `computed_register_soffset_is_never_captured`
* `vertex_stage_shift_maps_shader_registers_to_hardware_user_data_slots` —
  Avatar's and GTA V's measured shapes, shifted vs unshifted

*Recompile → SPIR-V, spirv-val (Vulkan 1.3)* (`recompile.rs`)
* `register_soffset_with_offset_materializes_a_resolved_capture`
* `unresolved_register_soffset_with_offset_refuses_naming_format_and_width` —
  all five widths
* `sload_dword_x1_materializes_instead_of_failing_the_shader` — asserts the
  pre-change behaviour still holds without a capture
* `eud_raw_window_records_and_refuses_an_unresolved_dynamic_offset`
* `full_chain_register_soffset_bytes_to_validated_spirv` — ISA words → parse →
  capture → recompile → assemble → spirv-val

`register_soffset_sload_dwordx8_is_named_refusal_not_panic` (pre-existing) was
updated: the refusal is still named, and now additionally carries the width,
format, base and soffset register.

naga is not the gate for this crate (documented false negatives for
`OpImageGather` and storage-image writes); `spirv_val_ok` is.

`dispatch_table_counts`: **353 → 358 rows**, **345 → 350 implemented**
(+5 three-operand rows).

## 7. Verification

* kyty-graphics **524 lib + 6 integration** green
* raeen-gpu **296 lib + all integration suites** green
* raeen-hle **567** green (untouched)
* `cargo fmt --all -- --check` clean
* `cargo clippy -p kyty-graphics -p raeen-gpu --all-targets -- -D warnings` clean

Note on stated baselines: the brief quoted kyty-graphics 509 lib and raeen-gpu
294 lib. The measured pre-change counts on `main` in this worktree are **510**
and **296**; no tests were removed.

One pre-existing flake observed: `raeen-gpu`
`draw_translate::tests::mip_chain_reads_mip_zero_from_the_end_of_the_allocation`
failed once in a parallel run and passes in isolation and in every subsequent
full run. It asserts deltas on process-global atomic counters
(`MIP_CHAIN_TEXTURES` et al.) that other tests also move, so it is
order-dependent. Unrelated to this change (no file it touches is in the diff).

## 8. Not claimed

No title was re-run. This is a decode / translate / wiring and test claim, not a
rendering claim for Avatar: Frontiers of Pandora, GTA V or ASTRO.BOT. In
particular:

* whether Avatar's and GTA V's vertex shaders now translate depends on the live
  user-data pointer being non-zero and the target dwords being readable at bind
  time — provable only by re-measuring;
* ASTRO.BOT's three compute shaders now parse; whether they *translate* depends
  on whether their soffset registers fall in one of the two resolvable shapes.
  If they do not, the outcome moves from a parse refusal to a recompile refusal
  that names the exact form — better diagnostics, same skipped dispatch. That is
  the honest ceiling of this change without a general scalar interpreter.

## 9. Follow-up worth its own task

A bounded **scalar abstract interpreter** (SharpEmu has one:
`Gen5ShaderScalarEvaluator` tracks the whole SGPR file plus a
`runtimeScalarRegisters` set). Raeen resolves two shapes by hand; an interpreter
would fold arithmetic chains (`s_lshl_b32` / `s_add_u32` off a live-in) and
subsume `resolve_scalar_soffset_bytes`, `pc_relative_base_address` and
`add_const_and_reg`. That is the correct home for anything this batch refuses.
