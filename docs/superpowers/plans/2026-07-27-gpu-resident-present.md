# GPU-resident present ("swapchain proper")

**Goal:** a guest frame never crosses to the CPU between the title's last draw
and the screen — render → (plugin pass) → composite → present, all
GPU-resident. This is the single biggest runtime-speed item after M3 and the
prerequisite for driving real GPU upscalers through plugin ABI v2.

## Where the frame goes today (and what each hop costs)

Runner child (Vulkan, offscreen) →
1. `vkCmdCopyImageToBuffer` + fence + **map/copy into `RenderedImage` (CPU)**
2. optional **CPU present plugin** (extra full-frame copy unless identity)
3. **memcpy into the frame-IPC shared slot** (double-buffered, seqlocked)

Shell (egui/wgpu) →
4. **memcpy out of the slot** into an `Arc<RenderedImage>`
5. upload to a wgpu texture
   *(now one direct `write_texture` — the `ColorImage` conversion copy was
   removed 2026-07-27; `egui_upload_us` in the PRESENT TIMING log measures it)*

At 1080p RGBA that is ~8.3 MB × 4–5 crossings per frame; at 4K ~33 MB each.
The readback (1) additionally serializes the GPU behind a fence.

## Done so far (2026-07-27)

- **Shell upload de-copied**: native wgpu texture registered with egui,
  written directly from the IPC frame bytes (no `ColorImage`, no per-frame
  allocation). Fallback to the old path when the backend is not wgpu.
- **Plugin ABI v2 landed** (`present_plugin::cabi`): `raeen_plugin_v2`,
  `RaeenHostContextV2` (Vulkan dispatch context), frame `kind`
  CPU-now/VULKAN-later, `RAEEN_CAP_GPU_FRAMES`, GPU output reserved.
  CPU-kind delivery is live; the contract does not change when GPU frames
  arrive — the host just starts setting `RAEEN_HOST_GPU_FRAMES`.

## Remaining phases

1. **Kill the child-side CPU hop into the slot** (keeps the process split):
   import the IPC mapping into Vulkan via `VK_EXT_external_memory_host` and
   `vkCmdCopyImageToBuffer` straight into the shared slot — readback and IPC
   copy become one GPU-side copy. Slot/seqlock discipline must move to the
   GPU timeline (write the sequence number after the copy's fence).
2. **Cross-process GPU sharing** (removes the CPU from the path entirely):
   child exports the color image (`VK_KHR_external_memory_win32`, keyed
   mutex or timeline-semaphore sync), Shell imports it into egui's wgpu
   device (`wgpu-hal` Vulkan `create_texture_from_hal`). The IPC header
   stays for sequencing/timing; pixels never leave VRAM. Driver-matrix
   testing required (AMD iGPU first — the dev machine).
3. **Plugin pass goes GPU**: with (2), set `RAEEN_HOST_GPU_FRAMES`, hand v2
   plugins the `RaeenVulkanImage` + context, take their `produced_image`.
   In-tree spatial upscalers (raeen-upscale) become compute passes — the
   GPL-clean reference proving the slot, as `nearest` does for v1.
4. **Motion vectors / depth** (the moat, gated on M4/M5 titles rendering):
   PM4-side extraction fills the `depth`/`motion` planes v1/v2 already
   reserve. External overlays guess motion; Raeen owns the command stream.

**Ordering guard:** none of this jumps the M-queue — phase 1–2 are
M4-adjacent infrastructure; phase 3–4 are gated on titles actually rendering.
