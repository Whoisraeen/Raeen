# M5 acceptance — Minecraft (Bedrock, retail PS5)

**Gate (from CLAUDE.md):** *One 3D title produces recognizable frames
(glitches OK); shader MVP for that title.*

**Status: MET**, on the evidence below. The runtime evidence is the same
recorded iron run as `docs/m4-acceptance-minecraft.md` (2026-07-27, user-owned
retail disc/PSN title — no Sony keys, SDK, or firmware in the repository; the
title is not redistributed and is not in-tree). This record adds **no new
runtime evidence**: it evaluates what that session already proved against the
M5 clauses, plus in-tree shader-path tests that need no retail content.

Recorded 2026-07-27. Host: Ryzen 5 7640HS (12 threads), Radeon 760M iGPU,
Windows 11 Pro, Vulkan 1.3.

---

## Same title as M4 — reasoning

The gate text names no title and does not require a title distinct from M4.
The milestones gate **capabilities**, not a per-title quota: M4 gated the
commercial-title UX loop (boot → interactive menu → saves → actionable logs),
M5 gates 3D rendering plus a working shader path. Minecraft happens to clear
both: it is a fully 3D title (perspective camera, textured voxel terrain,
depth-tested world) whose in-world rendering exceeds M4's 2D-menu floor and
lands squarely on M5's clause.

Supporting (not gate-bearing) second 3D data point: GTA V launches under the
same runtime (103 MB image, 612 trampolines, **presents frames**) before
tripping its own UD2 assert on AGC breadth — 83 missing `libSceAgc` NIDs,
dominated by async-compute (ACB) families Minecraft never touches. See
`docs/gta5-blocker-analysis-2026-07-27.md`. That is a scope record, not
recognizable-frames evidence; the M5 claim rests on Minecraft alone.

## Reproduce

1. Install the title so `<game folder>/eboot.bin` exists and add the parent
   directory in **Settings ▸ Game Folders**.
2. Launch from the Home rail; wait ~45–60 s for the menu (first run is slower:
   shader translation populates the on-disk cache).
3. `Play` → select or create a world → enter it.

## What must be observed

| # | Gate clause | Observation (recorded 2026-07-27) |
|---|---|---|
| 1 | Named 3D title | Minecraft (Bedrock edition, retail PS5). 3D world: perspective terrain, trees, water, depth-correct geometry. |
| 2 | Recognizable frames (glitches OK) | In-world scene renders with **real block textures** (not flat colour), HUD (hearts, hunger, hotbar, Emote button prompt), at 56 FPS. "Recognizable" is unambiguous: it looks like Minecraft. |
| 3 | Shader MVP for that title | The title's own guest shaders run through the Gen5 (AGC/`next_gen`) path in `kyty-graphics`: GCN/RDNA bytes → `shader/analysis.rs` input analysis → SPIR-V translation → Vulkan pipelines, with an on-disk pipeline cache. Decisive per-title fix below. |

## Clause 3: the shader MVP, demonstrated

The chain from "shader path broken" to "textures on screen" is documented end
to end:

1. **Symptom (from logs):** `guest shader analysis failed — draws binding it
   will be skipped stage="ps" addr=0x1700ab00 reason=… ps: direct sgprs`.
   Every Minecraft pixel shader carrying direct (raw-value) user SGPRs was
   refused, every draw binding one was skipped, and the world rendered as
   flat untextured blocks at 4 FPS.
2. **Fix (commit `d21e727`):** Kyty's PS4-era analyzer rejected direct user
   SGPRs only in the pixel stage; Gen5 VS never rejected them and Gen5 CS had
   already lifted the guard. The consuming push-constant plumbing
   (`shader_calc_binding_indices` → `push_constant_size`) is stage-agnostic.
   One guard change (`&& !next_gen`), legacy PS4 shaders keep the rejection.
3. **Effect (measured in the same session):** flat colour → real block
   textures, 4 → 56 FPS (compounding with the lock-parking fix `eef31c1`,
   which freed the CPU cores; `d21e727` stopped skipping the draws).

**In-tree evidence, no retail content required:**

- `input_info_ps_direct_sgprs_allowed_on_next_gen_rejected_on_legacy`
  (`crates/kyty-graphics/src/shader/analysis.rs`, added in `d21e727`) pins the
  Gen5 exemption and the legacy rejection.
- Full-chain shader tests `full_chain_vs_gcn_bytes_to_validated_spirv` and
  `full_chain_ps_gcn_bytes_to_validated_spirv` (kyty-graphics integration
  tests) prove guest GCN bytes → validated SPIR-V with no game present.
- Suite re-run for this record: `cargo test -p kyty-graphics` — **477 passed,
  0 failed** (472 unit + 5 integration).

## Honest limitations / known issues (gate requires these documented)

- **Evidence is the recorded 2026-07-27 session.** This acceptance record was
  written from that committed evidence (ledger + M4 record + commit
  `d21e727`); the retail title was **not** re-run for this record. Only the
  in-tree kyty-graphics tests were re-executed.
- **Recognizable, not correct.** Unknown PM4 context registers are skipped —
  mostly per-MRT colour-buffer descriptors and DCC/CMASK/FMASK compression
  metadata; MRT1–7 binding and fast-clear words are unimplemented. Rendering
  correctness was never verified against console output; deferred-rendering
  titles will need the MRT work.
- **Shader MVP is exactly that — an MVP scoped to this title.** It covers the
  shaders Minecraft actually submits (Gen5 VS/PS with SRT blocks and direct
  user SGPRs). The Gen5 analyzer still refuses many features (Kyty Gen5
  carries ~119 not-implemented exits generally), and GTA V already shows the
  next wall is AGC/ACB breadth, not this path.
- **Not a performance or playability claim.** 56 FPS at one location in one
  world on a light forward-rendered title; no soak test; a pre-fix long-session
  hang was observed and its absence after the fixes is unverified over long
  sessions.
- **Menu/pad input in the recorded run was synthetic** (`PostMessage` merged
  into guest pad state) — same code path as a real pad, but no physical
  DualSense in the loop.
- Audio, online/PSN, and most system dialogs remain HLE stubs.
