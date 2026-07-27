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

## Phase 1 groundwork — LANDED and measured (2026-07-27)

Two prerequisites are done, both tested:

- **The capability exists here.** `VK_EXT_external_memory_host` is detected
  and enabled on the logical device, and
  `minImportedHostPointerAlignment` is queried and exposed as
  `VulkanDevice::imported_host_pointer_alignment()`. **Measured on the dev
  machine's Radeon 760M: available, alignment 4096 B (one page).** Pinned by
  `crates/raeen-gpu/tests/external_memory_host.rs`, which passes (reporting)
  on devices without the extension rather than failing.
- **The IPC slots are now importable.** They were not: `HEADER_BYTES` was
  104, so every slot began at `104 mod 4096` and no slot pointer could ever
  be imported. The header is padded to a full page (**IPC version 5 → 6**;
  field offsets unchanged, only the padding after them grew), so slot 0 is
  page-aligned and — since `PIXEL_CAPACITY` is itself a page multiple — so is
  every later slot. Invariant pinned by
  `frame_ipc::platform::tests::every_pixel_slot_is_import_aligned`.

## The real obstacle to finishing phase 1 (measured, not assumed)

It is **not** Vulkan — it is frame ownership. `RenderedImage.pixels` is an
owned `Vec<u8>`, and **~143 sites across 8 files** read it: the present
plugin ABI (which hands `&[u8]` to out-of-tree plugins), the GPU-resource and
frame dumps, `last_image()`, the compute writeback path, the Shell's upload,
and the test suite.

If the GPU copies straight into the IPC slot, the runner never materialises
that `Vec`, so every one of those consumers needs a source. Two candidate
designs, neither trivial:

- **Borrowed frames** — make `RenderedImage` hold either an owned `Vec` or a
  borrow of the mapped slot. Touches every consumer's signature and adds a
  lifetime to a type that currently crosses an `Arc` and a process boundary.
- **Read-on-demand** — keep the copy, but only when a consumer actually asks
  (no plugin, no dump ⇒ no CPU touch at all). Smaller blast radius, and it
  preserves the fast path for the common case, at the cost of a branch and a
  cached-copy slot.

Read-on-demand looks correct: the *common* frame has no plugin and no dump,
so it should cost zero CPU copies, while a plugin frame pays exactly one.

**Do not attempt the remaining wiring as a quick change.** It is a
frame-ownership refactor with a live crash-isolation boundary underneath it,
and it wants its own session and its own soak test.

## Remaining phases

1. **Kill the child-side CPU hop into the slot** (keeps the process split):
   `vkCmdCopyImageToBuffer` straight into the imported shared slot — readback
   and IPC copy become one GPU-side copy. Slot/seqlock discipline must move to
   the GPU timeline (write the sequence number after the copy's fence).
   *Blocked on the frame-ownership decision above, not on any driver feature.*
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
