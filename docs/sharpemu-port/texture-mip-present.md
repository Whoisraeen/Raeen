# SharpEmu port — mipped textures read mip 0 from the wrong address

Date: 2026-07-28
Source: SharpEmu `6ee445f` (PR #470) "[AGC] Read mip 0 from its GFX10 mip-chain
offset", cross-checked against the live tip of
`reference/sharpemu/src/SharpEmu.Libs/Agc/GnmTiling.cs` and `Agc/AgcExports.cs`
(GPL-2.0-or-later, compatible with Raeen's GPL-2.0-only).

## The bug

`crates/raeen-gpu/src/draw_translate.rs::decode_texture` read the tiled source
for every swizzled T# at `t.base40()` — the descriptor base — and detiled it at
mip 0's extent.

On GFX10 an AddrLib mip chain is stored **smallest-first**
(`Gfx10Lib::ComputeSurfaceInfoMacroTiled`):

1. the small levels pack together into the **mip tail**, which occupies the
   FIRST swizzle block of the allocation;
2. the remaining levels follow in **decreasing** size;
3. **mip 0 lands at the END of the allocation.**

So any texture with `MAX_MIP > 0` was decoding the mip tail's bytes as if they
were a full-extent mip 0. SharpEmu measured this as scrambled menu text and
repeated icons.

* Wrong (before): `mip0_addr = base40()`
* Right: `mip0_addr = base40() + block_bytes + Σ mip_sizes[i]` for
  `i = first_mip_in_tail - 1` down to `1`, where each `mip_sizes[i]` is
  `align(w >> i, block_width) * align(h >> i, block_height) * bpp`.

### Worked example (the new unit test pins exactly this)

512x512 RGBA8, `SW_64KB_S` (SWIZZLE_MODE 9), `MAX_MIP = 9` → 10 levels.

| quantity | value |
|---|---|
| swizzle block | 65536 B → 128x128 elements, `log2(block) = 16` |
| levels the tail absorbs | `16 - 4 = 12` |
| tail extent | 128x64 elements (even `log2(block)` splits the extra bit into Y) |
| mip 0 (512x512) | too tall for the tail → own block grid, 1 048 576 B |
| mip 1 (256x256) | too tall → 262 144 B |
| mip 2 (128x128) | 128 > 64 → still too tall → 65 536 B |
| mip 3 (64x64) | fits → `first_mip_in_tail = 3` |

Memory layout, smallest-first:

```
base + 0x00000  [ mip tail block         65 536 B ]   <-- what was being decoded
base + 0x10000  [ mip 2                  65 536 B ]
base + 0x20000  [ mip 1                 262 144 B ]
base + 0x60000  [ mip 0               1 048 576 B ]   <-- what the sampler wants
                 total slice          1 441 792 B
```

`mip0 offset = 65536 + 65536 + 262144 = 393216 = 0x60000`. The decode was
393 216 bytes early — it was rendering the tail, at 4x the correct extent in
each direction.

`chain_slice_bytes = 1 441 792` matters separately: an **array layer** of a
mipped surface strides by its whole chain, not by mip 0's block grid, so the
per-layer read offset changed too.

### The fits-in-tail case

When the entire chain (mip 0 included) fits inside the tail block, mip 0 has no
block grid of its own: it is a **sub-rectangle of the detiled block**. The
sub-rectangle's element coordinate comes from the tail slot's offset
(`m << 8` for the first seven slots, `16 << m` beyond) Morton-de-interleaved
above bit 8 — odd bits give X, even bits give Y — then scaled by the 256-byte
micro-block extent. For 64 KiB at 4 B/element that is micro-block (8, 0) x 8x8
elements = element (64, 0).

Detiling the whole block first is load-bearing: in the tiled bytes mip 0's rows
are interleaved with the other tail levels, and only become contiguous once
linear.

## What was implemented

Pure math, unit-testable, next to the existing swizzle helpers:

* `crates/raeen-gpu/src/texture/tiling.rs`
  * `base_mip_placement(mode, elements_wide, elements_high, bpp_log2, resource_mip_levels) -> Option<BaseMipPlacement>`
    — the port of `GnmTiling.TryGetBaseMipPlacement`. Yields `byte_offset`,
    `chain_slice_bytes`, and `tail_element` for the fits-in-tail case.
  * `detile_mip_tail_base(...)` — detile the tail block, lift mip 0's
    sub-rectangle out of it.
  * `block_element_dimensions(mode, bpp_log2)` — port of
    `TryGetBlockElementDimensions`.
* `crates/raeen-gpu/src/draw_translate.rs`
  * `resource_mip_levels(max_mip, width, height, depth)` — port of
    `TextureDescriptor.ResourceMipLevels` / `GetMaximumMipLevels`: `MAX_MIP + 1`
    clamped to the most an extent of that size can carry. `MAX_MIP` describes the
    **allocation**; `BASE_LEVEL`/`LAST_LEVEL` describe one **view** of it, and
    sizing a chain from a view is wrong. The clamp is what stops a stale or
    all-ones `MAX_MIP` from computing an offset past the real allocation.
  * the swizzled tile-mode arm of `decode_texture` now reads at
    `base40() + mip0_offset`, strides array layers by `chain_slice_bytes`, and
    detiles through the tail sub-rectangle path when the chain fits the tail.

`kyty-graphics` already decoded `MAX_MIP` (`ShaderTextureResource::max_mip()`,
`crates/kyty-graphics/src/shader/resources.rs:279`); it had **no callers** before
this change. Raeen always captures a full 8-DWORD T# (shader analysis rejects
anything else), so `fields[5]` is always present — Raeen does not have SharpEmu's
`HasExtendedDescriptor` split, but it inherits the same underlying limitation:
if a title's `MAX_MIP` does not describe its allocation, nothing else in the
descriptor does either.

### Escape hatch

`RAEEN_NO_MIP_CHAIN=1` restores the read-at-descriptor-base behaviour, for A/B
bisecting a title whose `MAX_MIP` turns out not to describe its allocation. The
relocation is ON by default.

## What the mip warning now covers

`note_mip_view_base_level` previously returned early unless `base_level > 0`.
The **common** broken case is `base_level == 0` with `MAX_MIP > 0` — an ordinary
mip-0 view of a mipped texture — which took the tail's bytes with **no warning
at all**. It now fires for `max_mip > 0` too, with separate counters:

| counter | meaning |
|---|---|
| `MIP_VIEW_BASE_LEVEL_IGNORED` | `BASE_LEVEL > 0`: still served mip 0, still the wrong LOD (per-level addressing is not ported) |
| `MIP_CHAIN_TEXTURES` | `MAX_MIP > 0`: a mip chain was seen at all |
| `MIP_CHAIN_PLACEMENT_UNKNOWN` | a mip chain whose mip 0 could NOT be located — unsupported swizzle mode/element size, or a tail sub-rectangle that did not fit. These still read at the descriptor base, i.e. the pre-fix bytes. This is the counter to look at when a mipped texture still renders scrambled. |

The log line is rate-limited per `(base_level, last_level, max_mip)` triple; the
counters increment on every call.

## Honest limits

* **`BASE_LEVEL > 0` is still unimplemented.** This port locates mip **0**, not
  an arbitrary level N: level N needs the per-level tail slot, which
  `TryGetBaseMipPlacement` does not carry. Still counted, still warned, still
  served mip 0.
* **The fits-in-tail sub-rectangle is the least certain part of the port.**
  SharpEmu takes mip 0's tail slot to be the LAST slot (`m = maxMipsInTail - 1`),
  which is self-consistent with a smallest-first chain but is not obviously right
  when the chain is shorter than the tail capacity. It also puts the
  sub-rectangle half a block in, so any mip 0 wider than half the block cannot
  fit — reported as "no placement" (read at base, count it) rather than as an
  out-of-block offset. Both the port and the tests treat this as a **pin of the
  ported math**, not a claim about hardware.
* **Not verified against a title in this session.** The change is proven by
  in-tree fixtures only (a synthetic chain whose tail half is filled with a
  distinct byte, so serving the tail fails loudly). A retail A/B against
  Minecraft / GTA V is the remaining verification, and `RAEEN_NO_MIP_CHAIN=1` is
  there for exactly that comparison.
* **Linear (tile mode 0) mip chains are untouched.** SharpEmu gates the same way
  (`NeedsDetile`), and no linear mipped layout has been measured.
* Drive-by fix in the same lines: the CUBE per-face fallback detiled at **texel**
  extents while its destination slice was sized in **elements**, which for a BC
  cube produced a 16x-too-long buffer and would have panicked in
  `copy_from_slice`. It now detiles at element extents like the main path.

## Verdicts on the four unassessed SharpEmu commits

| Commit | Verdict |
|---|---|
| `ac883e4` (#473) logical width/height | **Already covered.** SharpEmu splits `LogicalWidth/Height` (guest-requested) from `Width/Height` (resolution-scaled backing) so a T# is not matched against a scaled host image. Raeen's `matching_live_target` (`draw_translate.rs`) already tries the guest extent first and then `scaled_sampling_extent(...)` against the same live-target list — the same split, expressed as two lookups instead of two field pairs. SharpEmu additionally accepts a *smaller* tiled texture against a larger logical target (`texture.Width <= guestImage.LogicalWidth`); Raeen requires exact equality. Loosening that is a deliberate aliasing-tolerance choice, unmeasured in Raeen, and NOT part of this bug — left alone. |
| `04557fd` (#447) refresh CPU-rewritten textures by write generation | **Applicable, out of scope, recommend its own task.** Raeen has no page-write tracking at all; its documented substitute is the per-bind `guest_sample_hash` re-probe, whose staleness window is spelled out in `draw_translate.rs` and which already names "Tier 5 page-dirty tracking" as the exact fix. Porting the generation counter without the tracker underneath it buys nothing. |
| `82ab181` (#550) `GuestImageWriteTracker` CPU sync on Windows | **Applicable, out of scope, same task as #447.** This is the tracker itself (689 lines, `src/SharpEmu.HLE/GuestImageWriteTracker.cs`): write-protect the range, fault handler marks dirty and restores write access, presenter consumes the dirty flag once per flip and re-arms. In Raeen this needs `raeen-runtime`'s VEH to cooperate, so it is a runtime + GPU feature, not a texture-decode fix. |
| `99004a3` (#649) host cached guest buffer | **Already covered.** SharpEmu's change is `preferredMemoryFlags: MemoryPropertyFlags.HostCachedBit` on the mapped guest-buffer mirror. Raeen already prefers `HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED` with a plain-host fallback for both the compute-buffer cache (`vulkan/cache.rs`) and the readback path (`vulkan/offscreen.rs`, which explicitly notes uncached readback being ~50x slower). The 2026-07-26 ledger entry already recorded SharpEmu host-cached buffers as covered. |

## Tests

`crates/raeen-gpu/src/texture/tiling.rs` (pure math, 7 new):

* `base_mip_placement_puts_mip_zero_at_the_end_of_the_chain` — the 512x512
  worked example above, plus the invariant
  `byte_offset + mip0 block grid == chain_slice_bytes`.
* `base_mip_placement_handles_the_4kib_block` — mode 5, 256x256, 9 levels →
  offset 90 112, slice 352 256.
* `base_mip_placement_refuses_instead_of_guessing` — single level, zero levels,
  unported mode, 32-byte elements, zero extent.
* `base_mip_placement_finds_the_in_tail_sub_rectangle` — (64, 0) at 4 B/el, and
  the slot scaling to (128, 0) at 1 B/el and (32, 0) at 16 B/el.
* `base_mip_placement_refuses_an_in_tail_rect_that_does_not_fit`.
* `detile_mip_tail_base_lifts_the_sub_rectangle` — functional: tile a whole
  128x128 block, lift the (64, 0) 64x64 rect back out; short block and
  out-of-block rect both refused.
* `block_element_dimensions_names_unsupported_inputs`.

`crates/raeen-gpu/src/draw_translate.rs` (2 new):

* `resource_mip_levels_clamps_max_mip_to_the_extent`.
* `mip_chain_reads_mip_zero_from_the_end_of_the_allocation` — the end-to-end
  acceptance test. A 128x128 RGBA8 mode-9 blob laid out
  `[tail 64 KiB of 0xA5][mip 0 tiled]` with `MAX_MIP = 3`; `decode_texture` must
  return mip 0's exact pixels, must count `MIP_CHAIN_TEXTURES`, must not count
  `MIP_CHAIN_PLACEMENT_UNKNOWN` or `MIP_VIEW_BASE_LEVEL_IGNORED`, and the same
  bytes with `MAX_MIP = 0` must come back as the all-`0xA5` tail (so the fixture
  provably reproduces the bug).
