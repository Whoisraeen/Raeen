# Rendering blockers + reference port plan (2026-07-22)

Three parallel audits: (1) in-tree rendering/FPS bug hunt, (2) Kyty/KytyPS5 port
inventory, (3) SharpEmu port inventory. Cross-verified; agents checked each
candidate against the current working tree.

**Corrections to prior assumptions:**
- `R_DMA_DATA` is **no longer unconsumed** — the ledger entry is partially stale.
  ACB form executes in `kyty-graphics/src/run.rs:811` → `cp_op_dma_data`; IT form
  (0x50) applied at submit in `libsce_agc.rs:1209-1247`.
- SharpEmu does **not** execute `IT_GET_LOD_STATS` or `IT_COPY_DATA` either —
  no port source there. The port source for those is **KytyPS5** `pm4Handlers.cpp`.
- `reference/kytyps5` is cloned (commit `8587638`, GPL-2.0 with Kyty MIT lineage
  retained) and is a richer PS5 source than Kyty for almost every target below.
  Add a `THIRD_PARTY_NOTICES.md` entry + flip `docs/reference-port-ledger.md`
  line 27 the moment the first KytyPS5-derived code lands.

---

## Tier 0 — the current crash (pause-menu poisoned-object fault)

The APR suspicion was half-wrong: records **do** execute — but three real bugs:

1. **Silent zero-fill on unresolvable file** — `libkernel.rs:599-611` +
   `libsce_ampr.rs:356-368`: when `appr_host_path(file_id)` is `None` (VFS
   resolve miss), the destination is zeroed and success reported, with **no log
   line naming the file**. An all-zeros LevelDocument produces exactly the
   `[r13+0xd0]`, `r13 = -0xd1` fault signature.
2. **Short-read bug** — one bare `file.read()` (`libkernel.rs:588`), no
   `read_exact` loop; tail of destination keeps whatever the guest had there.
3. **Completion never clears `ampr_write_offsets`** (`libkernel.rs:553-560`) —
   re-submit without explicit Reset re-executes stale ReadFile records into
   repurposed memory. Second independent heap-poisoning mechanism.

**SharpEmu model to adopt** (`AmprExports.cs:255-293`, `TryReadFileToGuestMemory`
:748-866, `KernelAprCompatExports.cs`): read the file **eagerly at
`AprCommandBufferReadFile` append time**, write `bytesRead` into the record
then, cache open host handles + positional reads (1 MiB pooled chunks); the
completion walk skips ReadFile records because data is already in guest memory.

**Immediate diagnostic:** log fileId + resolved-path-miss once per id on the
zero-fill path — this names the exact asset being lost on the next run.

## Tier 1 — self-inflicted PM4 holes (we emit packets we don't execute)

| Packet | Emitted by | Skipped at | Port source |
|---|---|---|---|
| `IT_DISPATCH_INDIRECT` (0x16) | `libsce_agc.rs:627,2063` | CP `run.rs` dispatch | KytyPS5 `pm4Handlers.cpp` |
| `IT_COPY_DATA` (0x40) | `libsce_agc.rs:2429-2438` | CP + submit | KytyPS5 `CpOpCopyData` L2207 (incl. **reference-clock writes** — games' GPU timestamps; Raeen has nothing) |
| `IT_INDIRECT_BUFFER` (0x3f) | `libsce_agc.rs:2686` (admitted in-code) | CP | KytyPS5 `graphicsRun.cpp` L595-723 — **buffer-stack** execution model + `CpOpBranch` L2135 |
| `R_VS` (0x01) / `R_PS` (0x02) full stage binds | titles using packet form | no `cp_op_nop` arm | KytyPS5 `pm4Handlers.cpp`; without it `vs_regs.data_addr` stays 0 → every draw skipped for packet-form titles |
| `IT_SET_PREDICATION` (0x20) / `IT_COND_EXEC` (0x22) | `libsce_agc.rs:400-404` | CP | KytyPS5 `CpOpSetPredication` L2078 + predicated-packet skip `graphicsRun.cpp` L680 + `CpOpCondExec` L2110 |
| `IT_GET_LOD_STATS` (0x8e) | `libsce_agc.rs:2820-2854` | CP | KytyPS5 `CpOpGetLodStats` L2039 (zero buffer + label=1 hack) |
| CE/DE counter family, `IT_WRITE_CONST_RAM`, reg-indirect forms | — | CP | KytyPS5 `pm4Dispatch.cpp` L14-191, L199-256 (142 register handlers vs Raeen ~96) |

Also: `IT_DRAW_INDEX_MULTI_AUTO` (0x30) / `IT_DISPATCH_DRAW` (0x8d) counted by
the decoder but never executed — the "559 draws" figure overstates reality.

## Tier 2 — draw-path correctness (blocks 3D rendering 100%)

1. **No depth attachment ever** — `draw_translate.rs:2116-2121` hard-codes
   `depth: None`; DB_* registers are latched (`run.rs:1651-1830`) but never
   consumed. Every 3D draw renders with no depth test → submission-order
   overdraw. Fatal for in-game correctness.
2. **MRT ignored** — only `render_targets[0]` bound (`draw_translate.rs:1934`);
   CB_COLOR1-7 latched but never attached.
3. **Cull/FACE winding risk with Y-flipped viewport** — code self-documents
   "culls EVERY primitive — silently and validation-clean"
   (`draw_translate.rs:2076-2083`); probe with `RAEEN_NO_CULL=1`.
4. Stale comment `draw_translate.rs:2126-2134` (indexed draws are real now) — delete.
5. **Scanout tiling/format skips → black frames** — `agc_exec.rs` skips
   unsupported tile modes ("never faked"). Port KytyPS5 **GPU compute detiler**
   (`tile.cpp` 1,644 + `gpuTiler.cpp` 537, recently perf-fixed) + Kyty's
   `TileTextureInfo_*.inc` tables.
   **CORRECTED 2026-07-24:** the claim "Raeen's `tiling.rs` is CPU-only, 2 modes"
   was already stale. It implements **4** modes (5/9/24/27) at 5 element sizes,
   and those equations were verified **bit-for-bit identical** to SharpEmu's
   (320/320 table entries; 8/8 simulated surfaces byte-identical) — see the
   2026-07-24 refresh in `docs/reference-port-ledger.md`. Still CPU-only, and
   still missing modes 1/4/8; a rate-limited `(tile_mode, format)` diagnostic on
   the refusal path now names which modes titles actually bind, so that port is
   gated on measurement rather than assumption. The bigger real gap here is that
   `texture_vk_format` has **no block-compressed (BC) format arms at all**.

## Tier 3 — ordered GPU side effects (correctness foundation)

SharpEmu's single most-repeated design rule (`AgcExports.cs:3726-3778`
`SubmitOrderedGpuSideEffect`): every guest-memory side effect (DMA_DATA,
WRITE_DATA, RELEASE_MEM label/timestamp, EVENT_WRITE) is enqueued on the GPU
queue so it applies **after** the draws that precede it in PM4 order. Raeen
applies all of them **eagerly at submit** (`libsce_agc.rs:1180-1290`) — labels
claim completion of GPU work that hasn't run; `sceVideoOutIsFlipPending` always
returns 0. Also adopt fail-open on side-effect faults (SharpEmu never fails the
submit; Raeen returns INVALID_ARGUMENT at :1188).

Companion pieces (`GpuWaitRegistry.cs` 483 lines + `MonitorGpuWaits`): produced-value
latching (label reset before re-check loses the wakeup — SharpEmu cites exactly
this for Astro Bot), CPU-store monitor thread, deadlock breaker.

ACQUIRE_MEM as ordered cache-invalidation point: `AgcExports.cs:3938-4010`.

## Tier 4 — FPS (5.4 fps → vsync-bound)

Measured ~185 ms/flip. Causes, in likely order of cost:

1. **Per-flip GPU→CPU readback + CPU re-upload** — `flush_and_present`
   (`agc_exec.rs:785-858`). SharpEmu model (`VulkanVideoPresenter.cs:5225-5340`):
   flip = GPU→GPU `CmdCopyImage` snapshot into an immutable per-version image;
   present reads only snapshots; **MAILBOX** present mode (`:4394-4431`); guest
   pacing via `PaceFlip` + free-running 60 Hz vblank thread
   (`VideoOutExports.cs:1291-1371`). Staged adoption: (a) GPU blit present,
   (b) snapshot-on-flip, (c) MAILBOX swapchain.
2. **Per-submit `wait_for_fences(u64::MAX)`** — `offscreen.rs:1185-1188,
   3495-3509`, `compute.rs:1159-1163`: full CPU↔GPU round trip per submission
   × 559 draws + 785 dispatches. KytyPS5 model: `commandScheduler.*` deferred
   flush, one submit per frame/window, fences/labels for completion
   (`Gpu::Process` `BufferFlush` only when progress). **Do not transliterate —
   redesign Raeen-native guided by it.**
3. **Per-flip HDR→sRGB re-encode, uncached** — `to_presentable`
   (`agc_exec.rs:2065-2091`): 8.3 M px × 3 powf per flip on 4K R16G16B16A16
   census-promoted frames. Cache keyed by (fb base, generation).
4. **Two full-frame clones per present** — `agc_exec.rs:788` (up to 66 MiB) +
   `:888` (33 MiB); store `Arc<RenderedImage>`.
5. **Synchronous compute per dispatch** — `compute.rs:31-60`: create/destroy
   staging+images, fence wait, full UAV readback, per dispatch. Reuse
   allocations; defer fence+readback to flush like draws already do.
6. Census fallback full-buffer scans (`agc_exec.rs:1297-1303` not sub-sampled;
   `FALLBACK_REELECT_INTERVAL = 64` forces periodic full flush).
7. Micro: `GuestDataPool`-style bulk DCB read window (SharpEmu
   `GuestDataPool.cs`); `guest_mem.rs:109-116` double-copy.
8. Load time (not flip rate): every HLE call serializes through global
   `CALL_LOCK` + guest-GIL (`dispatch.rs:129,293-361`) — caps boot throughput.

## Tier 5 — structural (after Tier 1-4 land)

1. **Page-watcher dirty tracking** — KytyPS5 `pageManager.cpp` 685 +
   `memoryTracker.cpp` 255 / SharpEmu `GuestImageWriteTracker.cs` 689:
   write-protect tracked guest pages, fault marks dirty, re-upload only dirty
   pages. Eliminates per-bind sample-hash re-validation. Raeen's VEH machinery
   transfers directly.
2. **VideoOut FlipQueue state machine** — KytyPS5 `videoOut.cpp` 1,725
   (Reserved→Recording→Ready→Presenting, EOP-sourced flips, backpressure).
3. **AGC packet-patching + fused-shader + interpolant APIs** — KytyPS5
   `agc.cpp` L589-1020, L1222-1264, L2354-3440 (~1,200 of 4,096 lines). Port
   per-NID as titles demand.
4. **PS5-only register handlers** — HTILE/DCC, NGG, border colors, VGT stages
   (~46 regs; mechanical batch port).
5. **Placeholder-based sparse VM** — KytyPS5 `kernel/memory.cpp` 4,091 (load
   times); only after the above.
6. pthread completion (KytyPS5 `pthread.cpp` 4,412, 152 ABI fns) + `libUlt.cpp`
   738 — incremental per-NID.

## Do NOT port / already done

- Shader recompiler: Raeen at **315/323 dispatch rows** (beyond Kyty's 204) —
  strongest area; only 8 NI rows + fetch shaders remain (reference: KytyPS5
  `recompiler/` + `shader.cpp` fused LS/HS).
- `GraphicsRender.cpp` → superseded by Raeen's own `vulkan/cache.rs`.
- SharpEmu loader fast-path / shader disk cache / HLE malloc: don't exist or
  Raeen is already ahead.
- `IT_GET_LOD_STATS`/`IT_COPY_DATA` from SharpEmu: not implemented there.
- `kyty-math`: orphaned (zero dependents) — wire or leave, don't extend.

## Suggested order of attack

1. APR: eager read + `read_exact` + clear write offsets + name-the-miss logging
   (unblocks current crash; small).
2. PM4 Tier 1 holes + reference clock (KytyPS5 `pm4Handlers.cpp`; same file,
   same test harness — land together).
3. Ordered side effects (Tier 3) — foundation for everything after.
4. GPU-side present + submission batching (Tier 4.1+4.2) — the FPS unlock.
5. Depth attachment + MRT + detiler (Tier 2) — 3D correctness.
6. Tier 5 structural phase.

---

## REGRESSION SAFETY — Minecraft (PPSA17221) is a live, rendering title

**Baseline (ledger, 2026-07-20/21, release builds, measured):** boots, renders
its boot-animation frames, peak animation window **60-63 flips/s**, p50 flip
interval 15.9-20.2 ms. Ceiling owner is the HLE vblank pacer
(`sceVideoOutWaitVblank` sleeps a fixed 16.667 ms, `libsce_video_out.rs:637`),
**not the GPU** (min interval 1.76 ms proves headroom). Later stalls at the
PSN-auth/menu gate — game-internal, not an emulator bug. Do not "fix" that.

**Hard rules for every change in this document:**

1. **Minecraft A/B is a gate, not a courtesy.** Before claiming any Tier 1-5
   item done: run Minecraft 180 s release and compare flips/s + wedge behavior
   against the 60-63 flips/s baseline, plus ASTRO.BOT. A Minecraft wedge or
   flip-rate drop = revert, same as the fire-and-forget precedent.
2. **Fire-and-forget flip flush is a KNOWN Minecraft killer (tried + REVERTED
   2026-07-20).** Minecraft wedged ~10 s in: flips stop, `pthread_sync`
   "stuck >3s" main-thread deadlock on a title mutex held by the flipping
   render-pool thread (3 threads spinning). ASTRO.BOT ran clean on the same
   build — the wedge is Minecraft-specific, cause never understood. The
   rendezvous flip flush at `present_scanout` was deliberately kept. **All
   Tier 4 present-path changes (GPU-side snapshot present, MAILBOX, async
   flip completion) must be opt-in env-gated and default-OFF for Minecraft
   until that deadlock is diagnosed.** SharpEmu's `SubmitOrderedGpuSideEffect`
   model is safe to adopt only if the flip rendezvous semantics are preserved.
3. **Open question to resolve BEFORE further perf work:** the stage-D GPU
   texture cache bundled into the M3 commit is UNVERIFIED — Minecraft after-run
   flips 43 → 22, build 969 → 59 µs (progress.md:2435-2437). Run the
   `RAEEN_NO_TEX_CACHE` A/B first; if it regresses, revert it. Do not stack
   new FPS work on top of an unverified one.
4. **Do not touch the vblank pacer semantics casually.** Minecraft paces its
   whole animation loop on `sceVideoOutWaitVblank`; the free-running 60 Hz
   vblank thread (SharpEmu `VideoOutExports.cs:1291-1371`) must keep the
   blocking-call contract for titles that import it, and 120 Hz ambitions
   must not change the 60 Hz default.
5. **PM4 Tier 1 arms must be additive-only:** execute packets that are
   currently skipped-by-length; never change the behavior of packets already
   executed (Minecraft's DCBs depend on the current register tables — proven
   byte-exact, progress.md:775).
6. **APR eager-read change:** Minecraft boots with 5 bundled .prx and heavy IO;
   verify identical boot (64 submissions baseline, progress.md:2106/2132) —
   eager reads must not alter read ordering or bytes visible to the guest.
7. Frame-dump comparisons (`RAEEN_DUMP_FRAMES`) for visual parity on the boot
   animation, not just flip counts.
