# Instruction coverage: MIMG `0x47` dmask, SMEM/SMRD `s_load_dwordx16`

Date: 2026-07-28 · Branch: `port-instr-coverage` (based on
`integration/sharpemu-sweep` @ `75c456e`)

Two measured GCN/Gen5 decode gaps, each the first shader blocker of a title on
the 9-title baseline taken at build `2e4bdca`.

| Title | Symptom on the baseline | Log line |
|-------|-------------------------|----------|
| ASTRO.BOT | stage `rendering`, 64 flips, clean exit | `unknown mimg format for opcode: 0x47 at addr <ADDR>, dmask: 0x2` |
| Avatar: Frontiers of Pandora | `timed_out`, 141 flips | `unknown smem instruction s_load_dwordx16, opcode = 0x4` |

Both are now decoded **and** translated end-to-end (parse → analysis →
recompile → spirv-val-clean SPIR-V). Neither is a rendering or milestone claim:
no title was re-run in this pass.

---

## 1. Opcode identities established

`reference/mesa` was **not** usable here: the local checkout carries only
`src/amd/{addrlib,common,registers}`, with no ACO/compiler tree, so it holds no
MIMG or SMEM opcode table (`grep -r gather4 src/amd` → nothing). Identity came
from the two translators that are actually upstream of this code.

### MIMG `0x47` = `image_gather4_lz`

- `reference/kyty/source/emulator/src/Graphics/ShaderParse.cpp:3132` —
  `case 0x47: KYTY_NI("image_gather4_lz");`
- `reference/kytyps5/src/graphics/shader/recompiler/ImageOps.cpp:166` —
  `{0x47u, "image_gather4_lz", Opcode::ImageGather4Lz, ImageSampleFlagLevelZero, 2u}`

The opcode was already wired in Raeen for `dmask 0x1`. The gap was the **dmask
model**, which KytyPS5's `DecodeMimg` states exactly:

```
inst.data_dwords = gather != nullptr ? 4u : CountDmaskComponents(inst.dmask);
...
if (gather != nullptr && !IsSingleDmaskBit(inst.dmask)) { /* unsupported */ }
```

So for the gather family the dmask is **not** a destination-component subset
(as it is for `image_sample`/`image_load`): it must be exactly one bit, it names
the single channel gathered, and the destination is always 4 dwords — one per
gathered texel. That single bit's index is precisely SPIR-V
`OpImageGather`'s `Component` operand, so all four single-bit masks are decided
by the encoding rather than guessed. All four are wired.

### MIMG `0x61` = `image_gather4h`

`reference/kytyps5/.../ImageOps.cpp:172` (`MIMG_GATHER_OPS`). Name only — see
"still refusing" below.

### SMEM/SMRD `0x04` = `s_load_dwordx16` (16 dwords / 64 bytes)

- `reference/kytyps5/src/graphics/shader/recompiler/MemoryOps.cpp:22` —
  `{0x04u, Opcode::SLoadDwordx16, 16, 32}` in `SMEM_OPS`
- `reference/sharpemu/src/SharpEmu.ShaderCompiler/Gen5ShaderTranslator.cs:1463,1486` —
  `0x04 => "SLoadDwordx16"`, and `:2441` — `"SLoadDwordx16" or "SBufferLoadDwordx16" => 16`
- `reference/kyty/.../ShaderParse.cpp:2424,2503` — `KYTY_NI("s_load_dwordx16")`
  in **both** the SMEM and the SMRD table (upstream never implemented either)

KytyPS5's `SMEM_OPS` is a single table consulted by both encodings, which is
what licenses filling in the legacy SMRD rows too (below).

*(SharpEmu was read from the current working tree, per the revert/restore
caveat: `6db095e` reverted work that `db4339f` later restored, so per-commit
reads mislead.)*

---

## 2. What now decodes and translates

### MIMG `image_gather4_lz`, all four single-bit dmasks

`crates/kyty-graphics/src/shader/parse.rs` — opcode `0x47` maps dmask
`0x1/0x2/0x4/0x8` to four formats, each with `dst.size = 4`.

`crates/kyty-graphics/src/shader/types.rs` — new `DMASK_4` format token and
three new formats `Vdata4Vaddr3StSsDmask{2,4,8}` beside the existing `…Dmask1`.

`crates/kyty-graphics/src/shader/recompile.rs` — the old
`recompile_image_gather4_lz_dmask1` body became
`image_gather4_lz_component(..., component)`, with four thin rows. The only
behavioral difference between them is one token:

```
%g4_r_<index> = OpImageGather %v4float %g4_si_<index> %g4_c_<index> %uint_<component>
```

`component` = the dmask bit index (0x1→0, 0x2→1, 0x4→2, 0x8→3). Everything
else — the 2D-only guard, the descriptor guard, the four texel stores into
`vdata..vdata+3` — is unchanged and shared.

### SMEM/SMRD `s_load_dwordx16`

`types.rs` — new format `Sdst16SbaseSoffset` (`DA16, S0A2, S1`), the
base+soffset sibling of the already-present `Sdst16SvSoffset` that
`s_buffer_load_dwordx16` uses.

`parse.rs` — SMEM `0x04` and SMRD `0x04` both produce
`SLoadDwordx16` / `Sdst16SbaseSoffset`, `src[0].size = 2`, `dst.size = 16`.

`recompile.rs` — the duplicated x4/x8 bodies were factored into
`sload_dword_wide(..., n, func)` (vertex-fetch-descriptor skip, gs-prolog
refusal, then `sload_dword_extended`), and `recompile_sload_dwordx16` is one
more caller with `n = 16`. `sload_dword_extended` was already generic in `n`,
and `ShaderEmbeddedConstantLoad::VALUES_MAX` was already 16, so the wide load
needed no new materialization machinery — only for the surrounding width
tables to stop being 1/2/4/8-only.

**Width tables extended (this was the real work — a parse-only fix would have
left the instruction decoding into silence):**

| File | Site | Effect if left unfixed |
|------|------|------------------------|
| `analysis.rs` | `find_scalar_load_bases` | EUD resolver never sees the load's base pointer |
| `analysis.rs` | `shader_capture_runtime_scalar_loads` | live SRT/user-data dwords never captured |
| `analysis.rs` | `shader_detect_embedded_constant_loads` | PC-relative constants never captured |
| `analysis.rs` | `eud_load_offset_for_register` | descriptor-alias rule blind to the load |
| `analysis.rs` | raw-EUD storage-buffer detection (×2) | dynamic V# tuples missed |
| `analysis.rs` | live-in EUD base scan | positional EUD fallback mis-ranked |
| `spirv.rs` | `shader_detect_eud_raw_window` | **raw window sized too small — dwords 8..16 read out of bounds** |
| `spirv.rs` | EUD raw-read constant seeding | undefined `%uint_N` ids in the emitted module |
| `recompile.rs` | `eud_load_into_reg_offset` | descriptor guard blind to the load |
| `recompile.rs` | MIMG descriptor-guard per-dword coverage | wide load could silently clobber a seeded index |

`recompile.rs::sgpr_dst_span` already handled `SLoadDwordx16`, and
`spirv.rs`'s `DetectFetch` already listed it — the type existed in the enum
with no producer.

### Legacy SMRD rows filled in

SMRD `0x00`/`0x01` (`s_load_dword`, `s_load_dwordx2`) were `KYTY_NI` although
their SMEM twins have shipped for a long time, and this parser already
normalizes both encodings to the same operand shape (SGPR base pair in
`src[0]`, constant byte offset in `src[1]`). They now reuse
`SdstSbaseSoffset` / `Sdst2Ssrc02Ssrc1`. SMRD `0x08` also gained the
`dst.size = 1` its SMEM twin sets.

---

## 3. Still refusing, and why

Nothing was bulk-added. Every remaining gap keeps a refusal that names the
specific opcode.

| Opcode | Name | Why not implemented |
|--------|------|---------------------|
| MIMG `0x47`, multi-bit dmask | `image_gather4_lz` | Illegal on the hardware (`IsSingleDmaskBit`). Refuses as `UnknownMimgFormat { opcode: 0x47, dmask }` — the dmask stays in the message. |
| MIMG `0x48`, `0x4f` | `image_gather4_c`, `image_gather4_c_lz` | Would need `OpImageDrefGather`, which has **no** `Component` operand — Vulkan always compares the depth channel. What AMD's single dmask bit selects for the compare is not established by any reference read here, and guessing it would silently sample the wrong channel. Named refusal retained. |
| MIMG `0x57`, `0x5f` | `image_gather4_lz_o`, `image_gather4_c_lz_o` | The GCN texel offset arrives in a **packed VGPR** (runtime), while Vulkan's gather `ConstOffset`/`Offset` operand must be a compile-time constant. No sound lowering established. Named refusal retained. |
| MIMG `0x61` | `image_gather4h` | No SPIR-V equivalent (horizontal gather). **Newly named** — it previously fell through to the generic `unknown_op` arm, so the log said only "opcode 0x61". |

---

## 4. Tests

TDD order per opcode: encode-bytes → expected decoded instruction, then a
full-chain test producing validated SPIR-V.

| Test | File | What it pins |
|------|------|--------------|
| `mimg_image_gather4_lz_all_single_bit_dmasks_decode` | `parse.rs` | 4 encodings → 4 formats, `dst.size == 4` for every dmask, full operand shape |
| `mimg_image_gather4_lz_multi_bit_dmask_refuses` | `parse.rs` | dmask `0x3` → `UnknownMimgFormat { opcode: 0x47, dmask: 3 }` |
| `mimg_image_gather4h_refuses_by_name` | `parse.rs` | opcode `0x61` refusal carries the string `image_gather4h` |
| `smem_s_load_dwordx16_writes_sixteen_sgprs` | `parse.rs` | SMEM `0x04` → `SLoadDwordx16`/`Sdst16SbaseSoffset`, `dst.size == 16`, 21-bit imm offset |
| `smrd_s_load_dword_x1_x2_x16_decode` | `parse.rs` | the three newly-filled legacy SMRD rows |
| `runtime_scalar_load_captures_sixteen_dwords` | `analysis.rs` | all 16 dwords captured from guest memory; `find_scalar_load_bases` reports `dwords == 16` |
| `image_gather4_lz_dmask_selects_the_gather_component` | `recompile.rs` | full chain × 4 dmasks: `%uint_<component>` in `OpImageGather`, 4 texel stores, **spirv-val clean** |
| `s_load_dwordx16_materializes_all_sixteen_dwords` | `recompile.rs` | full chain: raw-EUD window sized to `off/4 + 16`, 16 clamped reads, stores into `s16..s31`, **spirv-val clean** |

`spirv_val_ok` (Khronos spirv-val, Vulkan 1.3) is the gate for both full-chain
tests. `naga` cannot serve for either: its SPIR-V front end rejects
`OpImageGather` outright (`UnsupportedInstruction`) and rejects the
storage-buffer/array descriptor types real Vulkan accepts — both documented
false negatives in-tree.

`dispatch_table_counts` moved 348 → **352** rows, implemented 340 → **344**
(3 gather dmask rows + 1 `SLoadDwordx16` row), with the reason recorded in the
assertion messages.

**Counts:** `kyty-graphics` **502 lib** (was 494) + 5 integration;
`raeen-gpu` **294 lib** + 19 integration suites, all green.
`cargo fmt --all` clean; `cargo clippy -p kyty-graphics -p raeen-gpu
--all-targets -- -D warnings` green.

---

## 5. Not claimed

- No title was re-run. Whether ASTRO.BOT and Avatar advance past these
  blockers is unmeasured; the next blocker for each is unknown.
- Gather correctness beyond the encoding is untested against hardware: the
  hardware's (i0j1, i1j1, i1j0, i0j0) texel order is asserted to match
  `OpImageGather`'s, which is Kyty's/Vulkan's documented order but was not
  re-derived here.
