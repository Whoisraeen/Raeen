# Shader translation scorecard

This scorecard tracks the ordered shader-translation program separately from
the emulator-wide phases in `docs/codex-battle-ready-workflow.md`. A grade only
changes after its stated retail-free tests and eight-title compatibility gate
pass. The goal is exact translation when supported, safe emulation where that
is practical, and one classified refusal when support is genuinely missing.
Silent wrong output is never an acceptable fallback.

## Accepted baseline

- Report: `artifacts/compat/phase1-shader-corpus-c54cae1-dirty.json`
- Build identity: `c54cae135eb8+dirty` (release, `max-fps`)
- Protocol: eight titles, 180 seconds each, compared with
  `artifacts/compat/phase0-single-translator-rerun-44460ff-dirty.json`
- Result: 8 unchanged, 0 improved, 0 regressed
- Canaries: GTA V 12,512 flips and Minecraft 16,711 flips; both had zero
  shader errors and zero GPU errors

The accepted run followed a rejected Phase 1 capture candidate in which Avatar
reached 347 flips, a 32.2% drop outside the compatibility tolerance. Capture
was changed to avoid repeat filesystem opens and to preserve distinct binding
states, then the complete sweep was repeated. The accepted run reached 416
Avatar flips (18.8% below the Phase 0 baseline, inside tolerance) and 4,808
Subnautica flips. Both outcomes remain in the progress ledger; the passing run
does not erase the rejected measurement.

## Current grades

| Area | Grade | Evidence and next gate |
|---|---:|---|
| Translator ownership | A+ | Shader Phase 0 is green. Commercial shaders have one reachable owner: `raeen-gpu::shader_fetch` routes to `kyty-graphics`. The unused no-op-capable prototype was deleted and an architecture regression test prevents its return. |
| Failure corpus and clustering | A+ | Shader Phase 1 is green. A fresh eight-title sweep produced 139 content-addressed shader binaries, 3,066 exact binding-aware replay cases, and 48 stage/opcode/form clusters ranked by title fan-out. Warm offline replay processed every case in 2.009 seconds; a second strict replay reported zero regressions. Translation failures carry their exact analyzed resource ABI, and malformed or hash-mismatched corpus data fails closed. |
| Instruction decoding | C | Real Avatar and Subnautica shaders still refuse documented instruction/encoding forms. The accepted sweep recorded 17,053 Avatar and 546 Subnautica shader errors. Phase 2 must reduce Avatar to fewer than 100 unique clusters and Subnautica to zero. |
| Instruction lowerings | B+ | The production translator has 728 passing package tests, but real-title gaps remain, including three-component buffer operations, half/SDWA forms, LDS, image/mip forms, dynamic scalar offsets, gathers, and remaining vector ALU forms. |
| Descriptor discovery | C- | Static heuristics and placeholders remain. Phase 3 requires SGPR provenance plus draw-time resolution and zero silent placeholder substitutions in the eight-title sweep. |
| Wave/control-flow semantics | B- | Structured control-flow support exists, but Phase 4's complete EXEC/VCC/SCC and wave32/wave64 semantics matrix does not. |
| Image semantics | C- | Common sampled/storage paths work, but the full dimension, sampling, gather, atomic, format, MSAA, and mip-view matrix is incomplete. |
| Stage/interface completeness | D+ | The commercial pipeline currently proves vertex, pixel, and compute paths. Observed NGG/tessellation/geometry demand and construction-safe cross-stage interfaces still require Phase 6 evidence. |
| Differential and multi-driver proof | C | SPIR-V structural validation is strong, but decoder fuzzing against an external clean-room oracle, Vulkan execution comparison, and AMD/NVIDIA/Intel validation have not passed Phase 7. |

Overall shader grade remains **C+ (6.3/10)**. Phase 1 changes the speed and
trustworthiness of finding and fixing gaps, not the set of retail shaders that
translate, so instruction, descriptor, wave, image, and stage grades do not
move yet.

## Ordered gates

1. **Green — Phase 0:** exactly one translator reachable from the commercial
   pipeline.
2. **Green — Phase 1:** `cargo xtask shader-corpus` capture, ranked report, and
   exact-ABI seconds-scale offline replay from the eight-title corpus.
3. **Next — Phase 2:** Avatar under 100 unique failure clusters, Subnautica at
   zero, Minecraft/GTA V still zero shader and GPU errors.
4. **Phase 3:** real descriptor dataflow and zero silent dummy substitutions.
5. **Phase 4:** table-driven exact wave-semantics coverage for every
   implemented opcode.
6. **Phase 5:** generated image-operation matrix with no decode-reachable
   unimplemented cell.
7. **Phase 6:** every stage observed in the corpus translates and interfaces
   share one construction source of truth.
8. **Phase 7:** stated-size decoder differential fuzzing has zero disagreements,
   execution comparisons pass, and the suite is driver-configurable.
