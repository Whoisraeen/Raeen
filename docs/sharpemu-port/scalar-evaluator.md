# SharpEmu Gen5 scalar evaluator → `kyty-graphics`

**Date:** 2026-07-28
**Reference:** `reference/sharpemu` live working tree of `origin/main`,
`src/SharpEmu.ShaderCompiler/Gen5ShaderScalarEvaluator.cs` (2391 lines), with
`Gen5ShaderIr.cs`, `Gen5ShaderTranslator.cs` for the operand/encoding model.
Individual op semantics cross-checked against KytyPS5.
**Landed in:** `crates/kyty-graphics/src/shader/scalar_eval.rs` (new),
`crates/kyty-graphics/src/shader/analysis.rs`,
`crates/kyty-graphics/src/shader/mod.rs`,
`crates/kyty-graphics/src/shader/recompile.rs` (tests only).

> Read the live tree, not history. The decoder was developed on a
> `shader-decoder-part1` branch under `src/SharpEmu.Libs/Agc/` and relocated to
> `src/SharpEmu.ShaderCompiler/` on merge — the branch copy is ~24 days stale.
> A revert (`6db095e`) was later restored by `db4339f`.

---

## The problem this closes

An RDNA2 scalar memory load addresses `base_pair + SGPR[soffset] + imm`, all in
bytes, dword-aligned by truncation (established in the previous batch,
`docs/sharpemu-port/smem-register-soffset.md`). The recompiler can only lower
such a load if the **whole** address is a dispatch-time constant, so the soffset
register has to be proven at translate time.

Before this batch `analysis::resolve_scalar_soffset_bytes` could prove exactly
two shapes:

1. the register is never written *in the walked prefix* and names a captured
   live-in user-data slot;
2. the last writer before the load is `s_mov_b32` / `s_movk_i32` from a literal.

Everything else — i.e. **every** computed offset — became the named
`unresolved register soffset` refusal, which means `sload_dword_extended`
returns an error, the shader does not translate, and the draw or dispatch is
skipped. That is missing geometry, and the previous batch recorded the general
evaluator as its own named follow-up.

## What was ported

`scalar_eval.rs` is a behavioral re-implementation of the reference's
per-opcode semantics:

| SharpEmu | Ported as |
|---|---|
| `TryExecuteScalarAlu` L1118-1575 | `fold_sop2`, `fold_sop2_64`, the `s_mov`/`s_movk`/`s_mulk`/`s_brev`/`s_getpc` arms of `step` |
| `TryExecuteSaveExecScalarAlu` L1577-1675 | `fold_saveexec` |
| `TryExecuteScalarCompare` L1742-1803 | `fold_compare` |
| `TryEvaluateScalarOperand` L2249-2302 | `ScalarState::read32` |
| `TryEvaluateScalarOperand64` L1677-1708 | `ScalarState::read64` |
| `WriteScalarPair` L1710-1732 | `ScalarState::write64` |
| `TryDecodeInlineConstant` L2304-2345 | already done by Raeen's `operand_parse`; only the **64-bit sign extension** of the negative inline range survives as a distinct rule in `read64` |
| `SignedAddOverflow` / `SignedSubOverflow` / `ReverseBits` | same-named helpers |
| the `s_bfe_*` / `s_bfm_*` offset+width clamping | `bfe32`, `bfe64` |
| `TryEvaluate`'s ordered instruction walk L216-348 | `evaluate_before` |

### Folded opcodes

Scoped to what **Raeen's parser can actually produce** — implementing arms for
opcodes `shader_parse_sop{1,2,k,c}` refuses would be untestable dead code (see
"Not ported" below).

* **32-bit two-source:** `s_add_u32`, `s_sub_u32`, `s_add_i32`, `s_sub_i32`,
  `s_addc_u32`, `s_and_b32`, `s_or_b32`, `s_lshl_b32`, `s_lshr_b32`,
  `s_mul_i32`, `s_mul_hi_u32`, `s_bfe_u32`, `s_bfm_b32`, `s_cselect_b32`,
  `s_lshl4_add_u32`, `s_pack_ll_b32_b16`.
* **32-bit one-source / SOPK:** `s_mov_b32`, `s_movk_i32`, `s_mulk_i32`,
  `s_brev_b32`, `s_getpc_b64`.
* **64-bit pairs:** `s_mov_b64`, `s_not_b64`, `s_wqm_b64`, `s_and_b64`,
  `s_or_b64`, `s_xor_b64`, `s_andn2_b64`, `s_orn2_b64`, `s_nand_b64`,
  `s_nor_b64`, `s_xnor_b64`, `s_cselect_b64`, `s_lshl_b64`, `s_lshr_b64`,
  `s_bfe_u64`.
* **Save-exec:** `s_and_saveexec_b64`, `s_orn2_saveexec_b64`,
  `s_andn1_saveexec_b64` (the three the parser decodes).
* **SOPC → SCC:** all twelve `s_cmp_{eq,lg,gt,ge,lt,le}_{i32,u32}`.
* **Inert:** `s_nop`, `s_waitcnt`, `s_barrier`, `s_version`,
  `s_inst_prefetch`, `s_sendmsg`.

## The deliberate deviation: lattice, not concrete interpreter

SharpEmu runs a **concrete** interpreter. It allocates `uint[256]`, seeds it
from live user data, seeds `exec = RdnaWaveMask` and `vcc = 0`, executes, and
returns `false` with an error string for anything it cannot execute. Every
register it never seeded reads back as `0`.

Raeen evaluates a **two-point known/unknown lattice** instead
(`ScalarValue::{Known(u32), Unknown}`), because the two failure modes are not
symmetric:

* saying **unknown** when the value was knowable costs one skipped dispatch —
  exactly today's behaviour;
* saying **`Known(x)`** when the real value is not `x` makes the caller read the
  wrong descriptor dwords out of guest memory and hand a bogus V#/T#/S# to
  Vulkan. This project already has a measured `VK_ERROR_DEVICE_LOST` from that
  class (the EUD-pointer comment in
  `shader_capture_runtime_scalar_loads_shifted`).

That asymmetry is stated in the module docs and enforced structurally, not
per-arm:

1. **Only two combinators.** Every fold goes through `ScalarValue::map` or
   `ScalarValue::zip`, both of which return `Unknown` if any input is
   `Unknown`. No arm can silently default a missing operand to zero.
2. **`kill_destinations` on every unmodelled instruction.** A scalar load, a
   VOPC writing an SGPR pair, an opcode with no fold arm — all have their whole
   destination span (and SCC, when they write a scalar destination) forced to
   `Unknown`. This is what makes a guest-memory-derived value stay unknown
   rather than reading back as a stale live-in.
3. **`exec` and `vcc` are never seeded.** At shader entry `exec` is a
   hardware-supplied lane mask; SharpEmu's all-ones is a guess it can afford
   and Raeen cannot. Consequence: `s_cbranch_execz` is always undecidable, and
   save-exec results are `Unknown` unless the shader first pins `exec` to a
   constant.
4. **Only captured user data is seeded.** `sgpr[i] = Known(value[i - shift])`
   for hardware slot `i - shift` in `0..min(count, SGPRS_MAX)`. Registers below
   the NGG `shift` (gs-prolog hardware values) and at/beyond `count` (tgid /
   tid / wave-id system registers) stay `Unknown`.

## Control flow

SharpEmu forks supplemental paths at forward `s_branch` for resource discovery
(`QueuePath` / `ScalarPathKey` / `visitedPaths`, L142-280) and accepts whichever
concrete state a path happens to produce. That is fine for *discovering* which
descriptors exist; it is not sound for *deciding one address*.

`evaluate_before` instead returns the state immediately before the target
instruction only when two gates hold:

1. **Determinism.** It follows a branch only when the condition is a proven
   constant (`s_branch` always; `s_cbranch_scc0/1` when SCC is proven;
   `s_cbranch_vccz/vccnz`/`s_cbranch_execz` only if `vcc`/`exec` somehow became
   proven). An undecidable condition, `s_setpc_b64`/`s_swappc_b64`, or
   `s_endpgm` before the target is a `ScalarEvalRefusal`, not a guess. Because
   the trace is unique, the target cannot be entered with any *other* register
   file.
2. **No cycle through the target.** Two checks: the walk refuses a branch it
   would take backwards, and a pre-pass refuses any branch **after** the target
   whose destination is at or before it. The second is necessary because the
   re-entering edge of a descriptor loop (`load; s_add; s_cbranch_scc1 back`)
   sits after the load, where the walk never reaches.

`ScalarEvalRefusal` names the wall — `UndecidableBranch`, `Loop`,
`IndirectBranch`, `Unreachable`, `StepBudget`, `BadIndex` — and is logged at
debug level from `resolve_scalar_soffset_bytes` so a skipped shader says *why*.

One relaxation worth noting because it looks like a guess and is not:
`s_cselect_b32`/`s_cselect_b64` with an **unproven** condition still folds when
both arms are proven and equal. Whichever way SCC goes the result is the same
value, so this is exact.

## Wiring

`resolve_scalar_soffset_bytes(instructions, at, user_sgpr, shift)` (was
`(prior, inst, …)`) now delegates to
`scalar_eval::resolve_sgpr_before`, which tries two routes in order:

1. **Never written anywhere in the program** → the PM4-latched live-in. Sound
   without any CFG reasoning, which is why it is first. This is strictly
   *stronger* than the old prefix-only check: the old one accepted a register
   that a later instruction writes, and the new one requires the whole program
   to leave it alone.
2. Otherwise the deterministic walk, then read the register out of the lattice.

Both existing call sites are updated: the live-user-data capture in
`shader_capture_runtime_scalar_loads_shifted`, and the PC-relative capture in
`shader_detect_embedded_constant_loads` (which passes `user_sgpr = None`, so it
now benefits from `s_getpc_b64` folding too). The recompiler's refusal message
in `sload_dword_extended` is unchanged.

## Behaviour changes to be honest about

* A soffset written by `s_mov_b32` **inside branchy code** used to resolve on a
  prefix scan that ignored control flow; it now refuses if the prefix contains
  an undecidable branch. That is a soundness fix (the move might not execute),
  but it is a refusal where there used to be a capture. It degrades to the
  named refusal, never to a wrong address.
* The analysis test `computed_register_soffset_is_never_captured` was replaced:
  its fixture (`s_lshl_b32 s4, s5, 2` with `s5` a live-in) is now *provable* and
  resolving it is the point of this batch. It became
  `computed_register_soffset_resolves_through_the_scalar_evaluator`, and the
  unprovable-case coverage it used to give is now carried by two stronger tests
  (`a_soffset_computed_from_guest_memory_is_never_captured`,
  `a_soffset_advanced_by_a_loop_is_never_captured`).

## Not ported (and why)

* **Opcodes Raeen's parser refuses.** `s_not_b32`, `s_min/max_i32/u32`,
  `s_xor_b32`, `s_andn2_b32`, `s_ashr_i32`, `s_bfe_i32`, `s_bfe_i64`,
  `s_bfm_b64`, `s_absdiff_i32`, `s_lshl{1,2,3}_add_u32`, `s_pack_lh/hh`,
  `s_mul_hi_i32`, `s_subb_u32`, `s_addk_i32`, `s_bcnt*`, `s_ff1*`,
  `s_bitset*`, `s_brev_b64`, `s_cmpk_*`, `s_bitcmp*`, and the remaining
  save-exec forms all fail in `shader_parse_sop{1,2,k,c}`, so a shader using one
  never reaches the evaluator. Adding fold arms for them would be untestable
  dead code; adding them to the *parser* is a separate change that also needs
  recompile dispatch rows (and would move `dispatch_table_counts`). **This is
  the obvious next increment** — the evaluator arms are ~5 lines each once the
  parser and a lowering row exist.
* **`TryExecuteScalarLoad` / the global- and buffer-memory binding recorders**
  (L1851-2248, plus `TryReadGlobalMemory`, `TryDecodeBufferDescriptor`). These
  are SharpEmu's *resource discovery*, which Raeen already does its own way
  (`shader_detect_buffers`, `shader_capture_runtime_scalar_loads`,
  `shader_detect_eud_raw_window`). Porting them would duplicate, not extend.
* **`TryResolveImageBindings` / `TryCreateVertexInputBinding` /
  `TryCaptureVertexInputData`** — same reason; Raeen's vertex-fetch and image
  paths are established.
* **The supplemental CFG resource-discovery fork** — deliberately rejected for
  address resolution, see "Control flow".

## Verification

* `crates/kyty-graphics` — **578 lib** (was 524) **+ 6 integration**, green.
  Net +54: **50** in `scalar_eval::tests` (one per folded opcode with a
  known→known / unknown→unknown pair, the lattice combinator laws, user-data
  seeding incl. the NGG shift, inline-constant 64-bit sign vs zero extension,
  the guest-memory and lane-dependent kills, SCC invalidation, and one per
  control-flow refusal), **2** new in `analysis::tests` plus one replaced (see
  above), **2** full-chain in `recompile::tests`.
* Acceptance (both directions, from real GCN instruction words through
  `shader_parse` → capture → `spirv_generate_source` → `spirv_run`):
  * `full_chain_computed_soffset_bytes_to_validated_spirv` — `s_lshl_b32 s4, s5,
    2; s_add_u32 s4, s4, 16; s_load_dwordx4 s[16:19], s[12:13], s4 offset:16`
    resolves and passes **spirv-val** (Khronos, Vulkan 1.3). naga is not the
    gate — documented false negatives in this crate.
  * `a_memory_dependent_soffset_keeps_the_named_refusal` — the same load with
    `s4` derived from an `s_load_dword` result still errors with
    `unresolved register soffset`, with every address it could have guessed
    mapped in the fixture so a silent fold would show up as a pass.
* `dispatch_table_counts` **unchanged at 358/350** — no new instruction types.
* `crates/raeen-gpu` — 308 lib + all integration suites green.
* `cargo fmt --all`; `cargo clippy -p kyty-graphics -p raeen-gpu --all-targets
  -- -D warnings` clean.
* `raeen-hle` 566/567: `libsce_np_trophy2::create_context_and_handle_write_
  monotonic_ids` fails under the parallel run and passes in isolation — a
  pre-existing shared-state flake in a trophy NID test, unrelated to this batch
  (no `raeen-hle` file is touched).

**No title was re-run.** This is a translate/analysis and test claim. Whether a
specific measured title's soffset case resolves depends on the actual shape in
that shader; the class that now resolves is "any soffset computed by the folded
opcodes from live-in user data or in-shader constants, along a statically
decidable path".
