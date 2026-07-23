# Reference Recon → Roadmap (KytyPS5 / shadPS4 / SharpEmu / Kyty → Raeen)

Generated 2026-07-22 from an 8-agent reconnaissance of the **latest** reference trees
(KytyPS5 fresh clone, shadPS4 `9f31e64`, SharpEmu `559b7f0`, Kyty `4733b7e`) vs. our
codebase. Full per-item detail (verbatim `concrete_steps` + exact reference file/line
refs) is in the workflow result; this is the durable execution plan.

## The one thing to internalize

**A swapchain does NOT fix fps.** ASTRO runs at ~0.4 fps because
`sceVideoOutSubmitFlip → present_scanout → consume_flush(wait:true)` **blocks the guest
synchronously behind ALL queued DCB submits drained by a SINGLE GPU worker** — each flip
inherits the full worker-drain + fence-stall latency. The 60 Hz pacer already works. The
biggest fps lever is **rank 7 (bounded fire-and-forget flip)**, not the swapchain (rank 18).
Logs are 92% call-trace-ring spam (rank 1). There is also a **security bug**: a Windows VFS
drive-letter/symlink sandbox escape (rank 3).

Sequence: **measure → cheap high-impact wins → the flip fix → descriptor/texture correctness
so 3D draws → the large structural pieces.**

## Quick wins (small effort, high impact — do first)

| # | Item | Cat | Files | Port from |
|---|------|-----|-------|-----------|
| 1 | Collapse `log_call_trace` 4096-entry ring into ONE debug event; lead with a distilled ERROR crash report | logging | dispatch.rs, logging.rs | shadPS4 one-record-per-event |
| 3 | **Close Windows VFS drive-letter/symlink sandbox escape** (fail closed, canonicalize+contain) | security | kernel/filesystem/mod.rs | SharpEmu e01092a |
| 4 | Accept padded source row pitch in guest image uploads (bufferRowLength) | graphics | raeen-gpu agc_exec/draw_translate | SharpEmu 0ae785c (Dead Cells) |
| 5 | `sceAgc *Cb/*Dcb GetSize` sizing-probe stubs (size in rax, no writes) | graphics | libsce_agc.rs | SharpEmu 74a5198 |
| 6 | `libSceAjm` Batch{Initialize,JobDecode,StartBuffer,Wait,Cancel} silence stubs | codebase | libsce_media.rs | SharpEmu 2272b9b/d3600c9 |
| 2 | Always-on per-flip timing HUD (measure before optimizing) | fps | offscreen/agc_exec/present.rs | — |

## FPS ladder (after measurement)

| # | Item | Impact/Effort | Port from |
|---|------|---------------|-----------|
| 7 | **Bounded fire-and-forget flip (frames-in-flight semaphore)** — THE lever | high / med | (own; deadlock-aware, cap 1–2) |
| 12 | Ring of N=8 command buffers w/ per-buffer fences (CommandScheduler) | high / med | KytyPS5 commandScheduler.cpp |
| 13 | Drift-compensated high-res vblank pacing (waitable timer + accumulator) | high / med | shadPS4 AccurateSleep/Timer |
| 27 | Cheaper guest-mem present pass + sticky flip-miss fallback (GPU unpack) | med / small | own (agc_exec present_from_guest_memory) |
| 18 | Real Vulkan swapchain + fenced frame pool + present modes | high / **large** | KytyPS5 swapchain.cpp + shadPS4 vk_presenter |
| 19 | Dedicated high-priority present/vblank thread + FlipQueue | high / **large** | shadPS4 driver.cpp PresentThread + KytyPS5 videoOut.cpp |
| 28 | Serialize VkPipelineCache to disk (cold-launch stutter) | low / small | shadPS4 |

## Graphics correctness (get real 3D titles drawing)

| # | Item | Impact/Effort | Port from |
|---|------|---------------|-----------|
| 8 | Move invalid-descriptor from translate-time refuse → draw-time null fallback (robustness2 already on) | high / med | shadPS4 GetSharp Null() |
| 9 | Read sampled mip 0 from GFX10 mip-chain tail offset | high / med | SharpEmu 6ee445f |
| 10 | Sample 2D array / cube / 3D textures with real layers | high / med | SharpEmu 25d741b |
| 11 | Bind all enabled render targets (MRT, render_target_mask) | high / med | Kyty GraphicsRender.cpp |
| 20 | Persistent GPU texture/buffer cache + page-fault dirty tracking + overlap aliasing | high / **large** | KytyPS5 memoryTracker + Kyty GpuMemory.cpp |
| 21 | Descriptor resolution by dataflow trace over flattened SRT/EUD (TrackSharp) | high / **large** | shadPS4 resource_tracking + flatten_extended_userdata pass |
| 22 | Async compute rings (MapComputeQueue / DingDong / N-queue arbitration) | high / **large** | Kyty GraphicsRun.cpp + KytyPS5 graphicsRun.cpp |
| 23 | Bink2 FMV via FFmpeg host backend (depends on rank 6) | high / **large** | SharpEmu Bink/* (559b7f0) |
| 29 | MSAA sample count + resolve; broader GFX10 tiling swizzle coverage | med / med | SharpEmu GnmTiling |
| 30 | GPU-EOP flips embedded in command completion (needs rank 19) | med / med | KytyPS5 sync.h + videoOut.cpp |

## Load speed / module load

| # | Item | Impact/Effort | Notes |
|---|------|---------------|-------|
| 14 | Rework VFS read hot path — no handle clone, no double copy, no global write lock; read into guest VA directly | high / med | ~27 s of ASTRO load is this |
| 15 | On-disk cache of decrypted+linked+syscall-patched image (keyed by input hash + build version) | high / med | ~800 ms cold → tens of ms warm |
| 16 | Prefilter the syscall-patch instruction decode (`memchr 0F 05`, or rayon per-segment) | high / med | ~446 ms of main-module link |
| 26 | One-pass stub-eligible NID collection + resolve-path string-churn cleanup | med / small | 42,826-export / 717k-reloc title |

## Logging quality

| # | Item | Impact/Effort |
|---|------|---------------|
| 1  | (above) collapse ring + lead with crash report | high / small |
| 17 | Duplicate-suppression log sink (collapse consecutive identical events) | high / med |
| 24 | Subsystem log categories/targets (`hle::pthread=off,hle::gpu=warn`) + named threads | med / med |
| 25 | Panic/terminate handler that flushes + labels the crash with a stable code | med / small |

## Clean-room note

Kyty/KytyPS5 = MIT; shadPS4/SharpEmu = GPL-2.0 — all compatible with our GPL-2.0-only tree.
Attribute every port in `THIRD_PARTY_NOTICES.md` and log it in `docs/reference-port-ledger.md`.
Nothing is taken from GPL-3.0 study-only sources (OpenOrbis toolchain, fpPS4).
