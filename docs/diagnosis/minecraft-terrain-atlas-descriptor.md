# Diagnosis: Minecraft terrain atlas T# — does static resolution fail, and where?

**Status:** DIAGNOSIS ONLY — no fix in this branch. A parallel session owns
`kyty-graphics` / `raeen-gpu` edits.

**Baseline:** clean commit `0ea66d0` ("feat(gpu): port SharpEmu Gen5 scalar
evaluator for scalar-load addressing"), measured live on 2026-07-28 against
the user-owned retail Minecraft Bedrock build (`PPSA17221`, v1.21.43), five
headless runs, three of them with temporary (uncommitted) tracing on every
descriptor-resolution failure path plus a per-PS bind-ABI summary. Offline,
every dumped pixel shader was replayed through the same in-tree analysis
(`crates/kyty-graphics/tests/diagnose_terrain_atlas.rs`, committed with this
doc). Runs reached a saved world (`Opening level` → `Player connected.` in
`logs/raeen.log`) and walked it (`ls_up`), with frame dumps confirming the
in-world image.

## 0. Headline result — the premise is disproven at this commit

The working hypothesis handed to this diagnosis was:

> the terrain PS's atlas T# is runtime-/SRT-bound; shader analysis cannot
> resolve it; the descriptor reads back as all-ones poison;
> `check_read_only_texture_type` rejects it; `decode_texture` serves a 1x1
> transparent-black dummy and the draw proceeds untextured.

**Measured at clean `0ea66d0`, that chain never fires.** With warn-level
tracing on every link of it — placeholder installs
(`shader_get_texture_buffer` unresolved arm), all-ones poison replacement
(`check_read_only_texture_type` / `texture_descriptor_is_unresolvable`),
runtime-scalar-load capture skips *and* successes
(`shader_capture_runtime_scalar_loads_shifted`), and dummy serves
(`decode_texture` `base40()==0` arm) — two full menu→world→walk runs
(380 s each) produced:

* placeholder T# installs: **0**
* all-ones-poison replacements: **0**
* 1x1 dummy serves: **0**
* skipped draws / translation failures / `dynamic-image-descriptor`
  refusals: **0**

while the in-world frames still show the symptom: terrain geometry rendered
as **flat, per-face solid colors** (block-appropriate colors — copper teal,
tuff gray, water blue, foliage green — with per-face lighting variation)
under a fully textured HUD (hearts, hotbar, hand, glyph text all correct).

The terrain material's whole resource ABI statically resolves. The per-PS
bind summary for the in-world three-texture material family (the terrain
shape, world `QHe3erDCKLk@P1` — the M4/M5 acceptance world) reads:

```text
ps addr=0x1704e000  eud_used=true  eud_base=28  eud_raw=false  embedded_loads=0
  textures: [s0  direct type=9 fmt=56 base=0x57f90000]
            [s8  direct type=9 fmt=56 base=0x8fff0000]
            [s16 direct type=9 fmt=56 base=0x5c310000]
  samplers: [s24 direct] [s32 extended] [s36 extended]
ps addr=0x17050200  (identical ABI)
```

Real 2D fmt-56 texture descriptors with sane guest bases; the SRT sampler
pointer pair at `s[28:29]` was recovered by the EUD resolver
(`read_extended_user_data`, strategy "scalar-load base pair"), the two
runtime-loaded S#s resolved as extended (content read through that pointer),
and `sload_dword_extended` lowers the `s_load_dwordx4 sN, s[28:29], imm`
sampler fetches through the push-constant descriptor-index mapping. Nothing
is poisoned, nothing falls back.

**Conclusion: at `0ea66d0` the flat terrain is NOT an atlas-descriptor
static-resolution failure.** The failure is downstream of a correctly bound,
real atlas texture — in sampled *content* or *UV/interpolation*, see §4.

The earlier same-day session that measured the poison→dummy chain live was
running main's checkout **mid-merge with uncommitted parallel edits to
`kyty-graphics`/`raeen-gpu`** — a different build state than clean
`0ea66d0`. That is the most likely reconciliation; this measurement is from
a clean, named commit with the evidence lines quoted above.

## 1. The runtime-bound load shape (instruction level)

Even though it is not the terrain blocker, the runtime/SRT descriptor shape
this diagnosis was sent after is real, present in this title, and worth the
record — it is the shape any remaining "one element gray" UI case takes.

All of Minecraft's Gen5 material pixel shaders share one ABI (verified by
offline replay of every dumped PS through `shader_parse_ps` +
def-use tracing; tool committed as
`crates/kyty-graphics/tests/diagnose_terrain_atlas.rs`):

* **T#s direct-resident** in low user SGPRs (`s[0:7]`, `s[8:15]`,
  `s[16:23]`), sampled with no in-shader load.
* **S#s runtime-loaded** through a live-in SRT pointer pair in `s[28:29]`:

  In-world PS `0x17116300`-family (69 instructions, three samples):

  ```text
  pc=0x0008  s_load_dwordx4 s[32:35], s[28:29], 0x00   ; S# 0
  pc=0x0048  s_load_dwordx4 s[8:11],  s[28:29], 0x10   ; S# 1
  pc=0x00a4  s_load_dwordx4 s[0:3],   s[28:29], 0x20   ; S# 2
  pc=0x002c  image_sample v[2:5],  v[6:8],   s[8:15],  s[32:35]  dmask 0xF
  pc=0x0080  image_sample v[8:10], v[5:7],   s[16:23], s[8:11]   dmask 0x7
  pc=0x008c  image_sample v[5:7],  v[11:13], s[0:7],   s[24:27]  dmask 0x7
  ```

* A menu-era family (Play/world-select screens, PS `0x1700b800` /
  `0x17008500`) additionally assembles a **T# itself from x4 pairs** through
  the same pointer:

  ```text
  pc=0x014c  s_load_dwordx4 s[0:3], s[28:29], 0x20     ; T# dwords 0-3
  pc=0x0170  s_load_dwordx4 s[4:7], s[28:29], 0x00     ; T# dwords 4-7
  pc=0x0234  image_sample v[1:4], v[5:7], s[0:7], s[24:27]  dmask 0xF
  ```

  Note the **register-order vs load-order inversion** (dwords 0-3 from
  offset 0x20, dwords 4-7 from 0x00): any future capture that assembles
  descriptors from split loads must assemble by destination register, not by
  load order. No dumped Minecraft shader fetches these with
  `s_load_dwordx8`; the compiler splits the fetch.

Both shapes are plain SMEM: live-in base pair, constant immediate offset, no
register soffset, no PC-relative base, no dynamic index.

## 2. How each mechanism actually behaves on this shape (measured)

* **EUD recovery** (`read_extended_user_data`, analysis.rs): recovers
  `s[28:29]` as the extended-data pointer for the in-world family
  (`eud_used=true eud_base=28` in the live summary) and reads the SRT table
  behind it, so the runtime S#s resolve as *extended* descriptors and
  `sload_dword_extended`'s push-constant mapping covers their loads. This
  path is what makes the in-world family fully resolve.
* **8-dword runtime scalar-load capture**
  (`shader_capture_runtime_scalar_loads_shifted`, analysis.rs ~L744):
  correctly skips loads whose base is the recovered EUD pointer (the ASTRO
  fix). Live, its captures fired only for VS vertex-descriptor x4 loads
  (`base_reg=14`, thousands per run) — never for the PS families, and no
  skip-path (bounds/null/unreadable/short) fired either. Its `dwords != 8`
  gate means an x4-assembled T# would never install a texture binding from
  this pass; that gate is latent, not the measured blocker.
* **`sload_dword_extended` embedded-constant materialization**
  (recompile.rs): when a per-PC snapshot exists, it stores the **raw guest
  dwords** into the destination SGPRs. `Spirv::write_local_variables`
  (spirv.rs) seeds captured descriptors' start registers with the rewritten
  descriptor-array index, and the MIMG bodies index the Vulkan arrays with
  `OpLoad` of those registers — so a raw materialization that lands on a
  captured descriptor's registers **overwrites the seeded index with raw
  guest data after the prolog**. Live this did not occur
  (`embedded_loads=0` on every PS), but no in-tree test covers
  capture→install→MIMG end-to-end, and any fix that widens the capture must
  close this clobber hazard first.
* **0ea66d0 scalar evaluator** (`scalar_eval.rs`): folds ALU-computed
  soffsets only; kills every `s_load`-defined SGPR by design; whole-program
  walk refuses on the first undecidable branch (offline on these shaders:
  `UndecidableBranch { pc: 64/76/104 }`). These loads carry no register
  soffset, so the evaluator is simply not in this path — it neither caused
  nor can fix this family. It did not change the terrain (consistent with
  the live verification that terrain was already failing for a
  non-descriptor reason).
* **Placeholder/poison rescue** (`shader_get_texture_buffer` unresolved arm,
  `check_read_only_texture_type`, `shader_synthesize_placeholder_sampled_texture`,
  `decode_texture` base-0 dummy): all confirmed wired and all confirmed
  **silent** in these runs — nothing needed rescuing.

## 3. Live-run evidence trail

* Runs: five headless `--run-eboot` runs. Navigation that reaches the saved
  world at this boot profile: `0:neutral; 60000:cross; 70000:down;
  75000:down; 80000:cross; 150000:ls_up` (the title needs its savedata under
  `savedata/PPSA17221-app/` — without the copied worlds + user settings the
  same script lands in new-user onboarding, measured on runs 1-2).
* World entry markers: `Opening level 'minecraftWorlds/QHe3erDCKLk@/db'`,
  `Player connected.` (runs 3-5; run 3 loaded `jDFPC--2HxM@`).
* Instrumentation (all uncommitted, reverted after measurement):
  * capture pass: per-skip-reason warns + per-snapshot warns
    (analysis.rs);
  * poison replacement + unresolved-placeholder warns (analysis.rs);
  * dummy-serve counter warn (draw_translate.rs `decode_texture`);
  * once-per-address PS bind-ABI summary (shader_fetch.rs `translate_ps`).
* Frame dumps (`RAEEN_DUMP_FRAMES` + `RAEEN_DUMP_FRAME_INTERVAL`) confirm:
  menus/HUD fully textured; in-world terrain flat per-face colors in both
  measured worlds at `0ea66d0`.
* No `T# carries a mip chain` warnings fired in the in-world window: the
  bound material T#s carry `base_level=0, MAX_MIP=0`, so mip-tail relocation
  (SharpEmu #470) is not implicated for these textures.

## 4. Where the terrain bug actually lives — next diagnosis, not this doc

With descriptors, samplers, format (56), and a single-level atlas all
resolving, per-face flat color with correct per-block color selection means
each face samples ~one atlas texel. Two candidate mechanisms, in likelihood
order:

1. **UV interpolation/attribute path**: if the PS's texcoord interpolant is
   effectively constant per primitive (SPI interpolator settings decode, or
   the VS attribute fetch collapsing per-vertex UVs), every fragment samples
   the same atlas texel — flat faces with correct tile colors and working
   per-face lighting, exactly as observed. Start at
   `shader_get_input_info_ps` interpolator settings and the VS fetch of the
   UV attribute stream (the x4 `base_reg=14` V# loads captured live are the
   vertex-descriptor fetches).
2. **Sampled content**: the atlas upload from `base=0x57f90000` (fmt 56,
   type 9) decoding to near-uniform tile content (tiling/swizzle mode or the
   `guest_sample_hash` upload cache missing a late guest write). Cheap probe:
   dump the decoded upload for that base during the in-world window and eyeball
   it.

The blank-gray search-field box on the Play screen was to be checked as a
"same gap" candidate — with the descriptor chain measured silent on those
screens too (zero placeholders/dummies while the Play screen was on-screen in
runs 3-5), it is **not** explained by descriptor resolution either, and
plausibly shares the §4 cause.

## 5. If/when a real unresolved-SRT case shows up — fix design

For a title/shader where a T# genuinely is runtime-bound beyond the current
resolvers (the x4-assembled-T# shape is the concrete in-title example if a
future world binds it outside the EUD-recovered window), the owner mechanism
is **`shader_capture_runtime_scalar_loads_shifted`** (analysis.rs) plus the
materialization contract in **`sload_dword_extended`** (recompile.rs):

1. Group per-PC snapshots by destination register range; when x4 snapshots
   cover a contiguous 8-register window consumed by a sampled MIMG T#
   (`src[1]`), assemble the 8 dwords in **register order** and
   install/replace the `bind.textures2d` descriptor at that start register
   (mirroring the existing x8 arm: `sharp_dword3_is_buffer` +
   `check_read_only_texture_type`); likewise install x4 S# snapshots
   consumed as `src[2]` into `bind.samplers`.
2. Close the clobber hazard first: in `sload_dword_extended`, never
   materialize raw snapshot dwords over registers covered by a captured
   descriptor (skip the store; `write_local_variables` already seeds the
   rewritten index).
3. Add the missing end-to-end test: ISA words → parse → capture (x4 pair
   through a live-in pointer) → recompile → assemble → spirv-val, asserting
   the MIMG body indexes the descriptor array with the seeded index.

Analysis re-runs on every bind and the cache key already covers descriptor
type/format and captured values (`shader_fetch.rs`), so no cache changes are
needed.

## 6. Reproduction (retail-free summary)

* Offline replay of any dump set:
  `RAEEN_SHADER_DUMP_DIR=<dir> [RAEEN_DIAGNOSE_FILE=<substr>] cargo test -p
  kyty-graphics --test diagnose_terrain_atlas -- --nocapture` — prints, per
  dumped PS, every SMEM load with its base-pair producer chain, the
  0ea66d0 evaluator's verdict on it, and each sampled MIMG's T# def chain.
* Live: run the title headless with `RAEEN_DUMP_SHADERS`, the §3 input
  script and savedata prerequisite; `Player connected.` marks world entry.
* All retail bytes stayed in gitignored/scratch locations; this doc carries
  instruction-level disassembly only.
