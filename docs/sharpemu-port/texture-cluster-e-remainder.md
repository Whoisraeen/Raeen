# SharpEmu port — cluster E remainder (texture/detile/present correctness)

Date: 2026-07-28
Source: `reference/sharpemu` live tip `0535783` (GPL-2.0-or-later, compatible
with Raeen's GPL-2.0-only). Every commit below was read from the live tip (the
`6db095e` revert / `db4339f` re-apply trap does not affect any of them — all
seven predate or postdate that window and none was later reverted; the four
commits after `5b602c0` on the tip are e1a3b92/8f94562/99004a3/0535783, none of
which touches texture decode).

Context: the user reports Minecraft (PPSA17221) in-game block textures still
broken; these commits were the prime suspects. Verdict summary first, evidence
per commit below, and an honest note on what the logs say the *actual* remaining
Minecraft blocker looks like.

## Verdicts

| Commit | SharpEmu PR | Verdict | One-line reason |
|---|---|---|---|
| `25d741b` | #471 array layers | **ALREADY-HAVE** | Arrayed sampling + per-slice chain-stride upload + one-layer arrayed fallback views all present (independent provenance: ASTRO.BOT array work + the #470/#476 ports) |
| `1f3963c` | #483 exact-XOR factoring | **ALREADY-HAVE** (+ test gap closed) | `pattern_axis_term` port cites 1f3963c; this session added the independent re-derivation test the port had skipped |
| `0ae785c` | #475 padded row pitch | **N/A (architecture)** | Raeen's linear T# path reads at the descriptor's real `pitch()`; the SharpEmu path that had to *guess* a pitch (byte-count-validated RT-seed uploads) does not exist in Raeen |
| `224a36e` | #476 array-upload OOM memo | **ALREADY-HAVE** | `ARRAY_UPLOAD_UNSUPPORTED` in `draw_translate.rs` cites 224a36e verbatim |
| `327018e` | #448 linear-float sRGB at present | **ALREADY-HAVE** | Ported into `agc_exec.rs` (5 citations, incl. the float-unpack + sRGB-encode helper and its test) |
| `db9b204` | #468 resolution scale + DPI | **ALREADY-HAVE / N/A** | Texture-relevant half (scale + logical-size alias lookups) exists as `resolution_scale` + `scaled_sampling_extent`; the DPI half is Avalonia-child-process-specific, no Raeen analogue |
| `5b602c0` | #620 detile-pass cache key | **N/A** | Fixes SharpEmu's GPU compute detile pass (`VulkanDetilePass`), which Raeen does not have; the invariant it repairs (cache key must carry type/depth identity) is already satisfied by Raeen's key |

Net: **no correctness port was owed from this cluster.** One test was ported
(see below) to close a genuine verification blind spot.

## Evidence per commit

### 25d741b (#471) — sample 2D array textures with real layers

SharpEmu's change had four behavioral parts; each has a Raeen counterpart:

1. **MIMG array address ⇒ arrayed image + (u, v, slice).**
   `crates/kyty-graphics/src/shader/spirv.rs` (`SampledDim`, ~line 2142):
   2DArray and Cube both declare `Dim 2D, arrayed = 1` and carry the layer as
   the third coordinate. Pinned by
   `recompile.rs::array_texture_emits_arrayed_2d_image_and_vec3_coords` and the
   cube twin.
2. **Host/shader agreement on the declared vs bound type.**
   `draw_translate.rs::texture_view_kind` + `SampledDim::from_texture_type`
   delegate to the SPIR-V generator's own classifier — the exact "both have to
   agree" rule #471 introduces (`IsArrayedImageBinding`), stronger than
   SharpEmu's because it is one function, not two kept in sync.
3. **Every slice read at the per-slice mip-chain stride, uploaded as one
   buffer.** `draw_translate.rs` swizzled arm: `layer_stride` =
   `chain_slice_bytes` when a chain exists, else the block-grid size
   (~line 2069), with the per-layer detile loop below it.
4. **Fallback / single-layer sources still bind a one-layer 2D array view.**
   The `SampledDim` doc block (~line 1668) states layer-count-independence
   explicitly ("TYPE-driven, NOT layer-count-driven"), including for the #476
   OOM fall-back.

The guest-texture gate no longer rejects array/3D/cube types (types 9/10/11/13
accepted). Not ported from #471 itself, but behaviorally equivalent or stronger
on every point.

### 1f3963c (#483) — exact-XOR factoring + independent detile test

The factoring (`pattern_axis_term`, per-column X term, hoisted Y term) was
already ported — `crates/raeen-gpu/src/texture/tiling.rs` cites #483/1f3963c at
the function and both call sites.

**What was missing:** #483's `GnmTilingDetileTests` pins the mode-27
2 B/element table against an *independently re-derived* mask table. Raeen's
round-trip tests tile with the inverse of the *same* table (self-consistent
even if a row were transcribed wrong), and its known-answer pins covered only
the 4 B/element row. The 2 B/element row of `RB_PLUS_64K_RENDER_X` therefore
had no independent check.

**Correction from a retail capture (2026-07-31):** this test proved that Raeen
matched SharpEmu's generic Navi/RB+ equation; it did not prove that the equation
was the PS5/Prospero equation. A captured Minecraft mode-27, 2048x1024 RGBA8
terrain atlas decoded into repeated stripes with that table. Replaying the same
immutable bytes through KytyPS5's independently written
`Gen5RenderTargetOffsetInBlock` equation produced a coherent atlas. Raeen now
uses the Prospero equation for all five element sizes, cross-checks every
coordinate in a 300x300 grid against the shift/mask form, and keeps the
tile/detile round trips as a separate inverse-consistency test. The production
mode-27 output and independent offline replay are byte-identical (SHA-256
`B2A92477761E3F7CF95B21488801CB22B81549683FC0F983610DE7DE710F328D`).

### 0ae785c (#475) — padded row pitch in guest image uploads

SharpEmu's bug lives in `VulkanVideoPresenter.UploadGuestImageInitialData`: a
byte-buffer upload path (fed by `ProvideRenderTargetInitialData`, which seeds
newly created guest images from CPU-written guest memory) that **rejected** any
buffer whose length ≠ tightly-packed `w*h*bpp`, dropping padded-pitch uploads
and leaving textures blank. The fix *guesses* the pitch from the byte count
(width rounded up to 8/16/32/64/128/256 texels).

Raeen has no such path, and the underlying need is met better:

* Sampled linear T#s: `decode_texture` tile-mode-0 arm reads at the
  descriptor's **real** `t.pitch()` (`u32::from(t.pitch()).max(width)`), trims
  to tight rows, and handles volume slice pitch — no guessing.
  `pitch()` decode is pinned in `kyty-graphics/src/shader/resources.rs` tests.
* Render-target seeding: Raeen seeds attachments from its **own prior
  readbacks** (`framebuffers` map), whose extent/byte-size match by
  construction — there is no byte-count-validated guest upload to mis-reject.

**Honest limit noted, not owed by #475:** Raeen never seeds a *newly created*
render target from CPU-prewritten guest memory (SharpEmu's
`ProvideRenderTargetInitialData`, the Chowdren fog-layer case). A title that
CPU-prefills an RT before its first draw would start from cleared in Raeen.
Separate feature, separate evidence needed; out of scope here.

### 224a36e (#476) — never retry overrunning array uploads

Ported previously: `draw_translate.rs::ARRAY_UPLOAD_UNSUPPORTED` (~line 1600)
cites the commit, implements the address memo, drops to one base layer up
front, and keys the cache under `layers == 1` exactly as described.

### 327018e (#448) — encode linear-float flips to sRGB at present

Ported previously into `crates/raeen-gpu/src/agc_exec.rs` (helper around line
3062 citing "#448 (327018e)", used at both present call sites, with a
regression test citing it at ~line 3449). Raeen's present is a CPU-side
readback/compose rather than a blit-through-sRGB-intermediate, so the encode is
performed in the float-unpack instead — same transfer-function semantics.

### db9b204 (#468) — internal resolution scale + DPI fix

Two unrelated halves:

* **Resolution scale with logical-size alias lookups** — Raeen already has it:
  `resolution_scale` in `AgcGpuSession` runtime config,
  `scaled_sampling_extent` in `draw_translate.rs`, and `matching_live_target`
  probing guest extent first then the scaled extent (the #473 verdict in
  `texture-mip-present.md` documents the same split).
* **DPI awareness of an isolated Avalonia child-process surface HWND** — no
  Raeen analogue; Raeen's Shell is egui (winit handles DPI) and frames cross
  processes as pixels (`frame_ipc.rs`), not as HWND geometry queries. N/A.

### 5b602c0 (#620) — detiled cache key for VulkanDetilePass

The commit adds `Type`/`Depth` to the identity of textures produced by
SharpEmu's **GPU compute detile** path so they cache under the same key the CPU
path uses. Raeen has no GPU detile pass (all detile is CPU-side in
`texture/tiling.rs`), and Raeen's persistent-texture cache key
(`texture_cache_probe`) already carries
`base/width/height/layers/depth/cube/array/volume/format` — the full identity
whose absence caused SharpEmu's per-draw re-detile. Nothing to port. (The
reference-port ledger reached the same verdict on 2026-07-2x; this confirms it
against the live tip.)

If Raeen later grows a GPU detile pass (worth it — SharpEmu measured CPU
detile at 568-879 ms/s on one title before memoization), #620 must be part of
that port from day one.

## What was breaking Minecraft's block textures

The previous conclusion below was refuted by better evidence. The table was
self-consistent and equivalent to SharpEmu, but wrong for the captured PS5
mode-27 atlas. The Prospero equation correction now yields recognizable grass,
trees, water, mobs, HUD, and lighting in the retail in-world frame. The older
shader/synchronization observations remain useful as historical evidence, but
they were not the cause of this texture scramble:

* **6x `guest shader translation failed — draws binding it will be skipped`,
  all `stage="vs"`, all `SLoadDwordx2 … s[14:15], <imm>`** — the SMEM
  register-soffset class. Skipped VS draws = whole geometry (and its textures)
  missing. The soffset port (`a192cf1`, 2026-07-27) landed *after* this log was
  captured; a fresh run may already look different.
* 143x unknown context register 0 writes and 75x suspended `WAIT_REG_MEM`
  (37 force-resumed) — sync trouble that can starve or misorder uploads.
* An older menu-era log additionally shows `texture_cap_skips=224` (draws
  refused by the 96 MiB per-stage texture cap, `RAEEN_MAX_STAGE_TEXTURE_MIB`) —
  if in-game runs still hit this, whole draws (blocks included) vanish legally.

The strict performance gate remains separate: the post-fix diagnostic retail
run still contained an 11.6-second no-flip window owned by guest streaming code.
Presentation and the corrected atlas decode are not evidence that this guest
critical section is fixed.

## Tests

* New: `raeen-gpu texture::tiling::sw_64kb_r_x_matches_kytyps5_prospero_equations`.
* Updated: `raeen-gpu texture::tiling::sw_64kb_r_x_2bpp_matches_an_independent_re_derivation` now uses the Prospero equation.
* Gate: `cargo test -p raeen-gpu` and `cargo test -p kyty-graphics` green (counts
  in the session report); `cargo fmt --all --check` clean.
