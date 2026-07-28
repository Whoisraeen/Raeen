# ASTRO.BOT: 128 frames → 0 frames, and the `0xffffffff` return that hid it

Measured 2026-07-28. Raw log:
`artifacts/compat/raw/baseline-1785273714952/PPSA21564-d1d2e7566901.stdout.log`
(report `artifacts/compat/public-launch-20260728.json`, build `60c63f9c1392`).

## What the run did

```
WARN raeen_runtime::arena:  guest fixed-address map failed; host regions in the requested range:
       requested=0x1000000000..0x10c8000000 failed_at=0x1000000000 title_window_reserved=true
         0x1000000000..0x1090ca0000 free
         0x1090ca0000..0x1090d44000 reserved private
         0x1090d44000..0x1090da0000 committed private
         ... (eleven more 1–2 MiB reserved+committed pairs, up to 0x1093000000)
         0x1093000000..0x10c8000000 free
WARN raeen_hle::libkernel: sceKernelMapDirectMemory: cannot map len=0xc8000000 at requested=0x1000000000
INFO guest: ASSERT: D:\asobi\6.0\source\engine\app\Module\Memory\DirectMemoryAllocator.cpp:122
INFO guest: sceKernelMapDirectMemory error 0xffffffff
ERROR raeen_runtime::dispatch: guest fault at 0x1000004a7df5 (read 0xffffffffffffffff)
```

The faulting bytes at RIP are `cd 41` — `int 0x41`. This is **not** a wild
dereference: it is the title's own assert trap, executed deliberately three
instructions after the failed map. Windows reports a software interrupt through
an unmapped gate as `STATUS_ACCESS_VIOLATION` reading `0xffffffffffffffff`,
which is why the fault line reads the way it does.

Timeline: the map fails 0.45 s after `_start`, before any draw. The title exits
at 4.0 s having presented 0 frames.

## Was this "forward progress exposing a new wall"? No — refuted.

The hypothesis was that today's shader merges let ASTRO translate shaders it
previously refused, so it runs further and reaches this call for the first time.
The log refutes it on two counts:

* The run **never reaches shader translation.** It dies at the engine's first
  direct-memory allocation, 20 ms after `main()` prints its banner. The log is
  179 lines and contains no `spirv:` / `shader` line at all.
* The call is **not new.** Comparing the four measured ASTRO runs:

  | run_id | build | flips | ASTRO log | `cannot map` present? |
  |---|---|---|---|---|
  | `baseline-1785235662044` | `2e4bdcae8a9a` | 64 | 3.0 MB | no |
  | `baseline-1785261005988` | `da37775af1dd` | 36 | 332 KB | no |
  | `baseline-1785264692261` | `36a9b18ccbb1` | **128** | 444 KB | no |
  | `baseline-1785273714952` | `60c63f9c1392` | **0** | 28 KB | **yes** |

  The HLE logs this call only on failure, so its absence from the earlier logs
  means it **succeeded** there. Same title, same request, three prior successes.

## Why the mapping fails

`0x1000000000` is a **second** fixed direct-memory base ASTRO maps at, distinct
from the `0x3_0000_0000` libc mspace base that `TITLE_VA_WINDOWS` already
defended. Only the first was claimed at startup, so the second was left on offer
to the host allocator — and eleven host **thread stacks** (the 1–2 MiB
`reserved`+`committed` pairs in the dump, created by the guest pthreads that
start 15 ms earlier) landed at `0x1090ca0000..0x1093000000`, inside the
`0x1000000000..0x10c8000000` the title asks for. The fixed `VirtualAlloc` then
cannot be placed and `GuestAllocator::map_at` returns `None`.

Nothing in the memory path changed between `36a9b18` and `60c63f9`. What changed
is how much the host allocates before a launch: that range is
`686d16e..60c63f9`, which is the GPU-plugin / Vulkan resource-cache / offscreen
rendering / upscaler series. More host allocation earlier moved where Windows
placed the thread stacks. **The mapping was always a coin flip; the GPU work
only changed which way it landed.** This is the same failure class as the
documented 2026-07-21 Shell-vs-CLI divergence at `0x3_0000_0000`, at an address
that was never added to the defence.

## Fixes

### 1. The window (this is what restores the frames)

`crates/raeen-runtime/src/arena.rs`. `TITLE_VA_WINDOW_MIN`/`_LIMIT` became a
list, `TITLE_VA_WINDOWS`, and `0x10_0000_0000` was added with the same
`PS5_DIRECT_MEMORY_SIZE` span as the first entry. `claim_title_va_window` walks
every entry at startup, so the host allocator is never offered either range.

`0x10_0000_0000` is also **Minecraft's V8 pointer-compression cage hint**
(`hinted_reservation_honors_the_guest_window_and_exact_unmap_releases_it`), so
claiming it naively would have traded ASTRO for the M4/M5 title: a
`MEM_RESERVE` over our own block fails with `ERROR_INVALID_ADDRESS`, V8 would
have been relocated off its hint to a non-4-GiB-aligned address, rejected it,
and retried — exactly the regression hint support was added to fix.
`reserve_with_hint` therefore now serves a reservation whose span lies inside a
claimed block **from that block, at the requested address**. The claim is *for*
the guest, not against it: it takes the address space from the host while still
handing it to the guest. The block is deliberately not recorded in
`os_reservations`/`external_mappings` (not this arena's to release);
`window_commits` carries the span so `Drop` decommits whatever demand-commit
later backed.

### 2. The return code (this is why the failure was unsurvivable)

`crates/raeen-hle/src/libkernel.rs` had `const HLE_ERROR: u64 = 0xFFFF_FFFF`,
described in its own doc comment as "not a real `SCE_KERNEL_ERROR_*` code — just
a nonzero value". A guest cannot classify it. Every guest-visible use is fixed
and the constant is **deleted**, so the class cannot come back:

| Export | Failure | Was | Now |
|---|---|---|---|
| `sceKernelMapNamedDirectMemory` / `MapDirectMemory` / `MapDirectMemory2` | fixed address unplaceable | `0xffffffff` | `SCE_KERNEL_ERROR_ENOMEM` (`0x8002000C`) |
| `sceKernelAllocateDirectMemory` | arena exhausted | `0xffffffff` | `SCE_KERNEL_ERROR_EAGAIN` (`0x8002000B`) |
| `sceKernelAllocateDirectMemory` | `physAddrOut` unwritable | `0xffffffff` | `SCE_KERNEL_ERROR_EFAULT` (`0x8002000E`) |
| `sceKernelMapFlexibleMemory` | arena exhausted | `0xffffffff` | `SCE_KERNEL_ERROR_ENOMEM` |
| `sceKernelMapFlexibleMemory` | `addrOut` unwritable | `0xffffffff` | `SCE_KERNEL_ERROR_EFAULT` |
| `sceKernelReserveVirtualRange` | no address space | `0xffffffff` | `SCE_KERNEL_ERROR_ENOMEM` |
| `__tls_get_addr` | TLS allocation failed | `0xffffffff` | `0` (NULL) |
| `__tls_get_addr` | offset past the bounded block | `SCE_KERNEL_ERROR_EINVAL` | `0` (NULL) |

The two `__tls_get_addr` rows are the mirror-image bug: that export returns an
**address**, not a status, so neither a sentinel nor an `SCE_KERNEL_ERROR_*`
belongs there — both are pointers a caller will dereference. `NULL` is the
convention `hle_mmap` already documents for an address-returning failure.

Codes taken from shadPS4 (`src/core/memory.cpp` `MemoryManager::MapMemory`
returns `ORBIS_KERNEL_ERROR_ENOMEM` when a fixed mapping cannot be placed;
`src/core/libraries/kernel/memory.cpp` `sceKernelAllocateDirectMemory` returns
`ORBIS_KERNEL_ERROR_EAGAIN`). Behaviour re-implemented in Rust; no code copied.

### 3. `MAP_FIXED` is now honoured (defence in depth)

`hle_map_direct_memory` read `args[3]` (`flags`) nowhere: it treated **every**
non-zero requested address as mandatory and hard-failed when it could not be
served. Orbis honours the address only under `MAP_FIXED`; otherwise it is a hint
and the kernel places the mapping where it can, reporting through `addrOut`
(shadPS4 takes its `SearchFree` path whenever `Fixed` is clear). An unservable
hint now falls back to publishing `direct_memory_start` — the same answer the
no-hint branch already gives, which keeps the mapping and the direct memory one
storage. This cannot regress a case that succeeds today, and it means the next
unforeseen fixed base degrades instead of killing a title.

## Will the 128 frames come back?

**Likely yes, and the reasoning is mechanical rather than hopeful** — but this
has not been re-measured on hardware, so it is a prediction, not a result.

* The map is the *only* thing that failed. Fix it and the run continues from the
  same point the 128-frame build did, on a build that is otherwise a superset.
* With the window claimed, the range is inside a reservation we own, so `map_at`
  takes its `in_title_window` `MEM_COMMIT`-only path. Even if Windows refuses a
  3.1 GiB up-front commit, that path degrades to demand-commit and still returns
  the address — so the call now succeeds under commit pressure too, where it
  previously hard-failed.

**The error-code fix alone would NOT have restored the frames.** The guest
asserts on *any* non-zero return: `DirectMemoryAllocator.cpp:122` prints the
value and traps regardless of which code it is. A correct `ENOMEM` would have
produced a better log line and the same 0 frames. It is still required — a guest
that cannot classify a failure cannot run its own failure handling — but the
window is what makes the call succeed.

Residual risk: 128 frames in 180 s is 0.7 FPS, so ASTRO was never healthy. Past
this wall it will meet whatever limited it before, and the frame count may land
anywhere near 128 rather than exactly on it.

## Related: GTA V's `s_buffer_load` refusal

Same commit, different subsystem. Measured
`artifacts/compat/raw/baseline-1785273714952/PPSA04264-e94fe9c5ee18.stdout.log`:

```
ERROR kyty_graphics::shader::spirv: Recompile_SBufferLoadDwordx8_Sdst8SvSoffset:
  not supported: no storage buffer bound for the V# and no resolved capture:
  SBufferLoadDwordx8 [Sdst8SvSoffset] x8 dwords, V#=s[20:23], soffset=none, imm=0x0, pc=0xd4
```

998 of that run's 999 shader errors are this family, across three vertex shaders
(`0x148d69c00`, `0x148d6a800`, `0x148d6b000`; the third is the x16 form on
`s[0:3]`).

The failing precondition was the **live-in guard** in
`shader_capture_vsharp_buffer_loads` (`crates/kyty-graphics/src/shader/analysis.rs`):
the pass required the V# quad to be an unwritten live-in user-SGPR quad. GTA V's
shaders load the V# themselves out of an SRT descriptor table —
`s_load_dwordx4 s[20:23], s[12:13], 64` — so the quad *is* written in-shader, the
capture was skipped, `buffers_num == 0`, and the recompiler refused.

The descriptor was knowable without guessing: the producing `s_load` is itself
already captured from guest memory by the pointer-load pass that runs
immediately before, so **its four dwords are the V#**. The pass now accepts
exactly two provenances — live-in, or a single earlier writer that defines the
whole quad at its base and whose own dwords are already captured. Two writers, a
partial or offset write, a quad assembled by moves, or an unproved producer all
keep the named refusal.

This adds no staleness the module does not already carry: the producing load is
materialized from that same snapshot, so the translated shader is pinned to
those dwords either way, and the shader cache key hashes every captured value —
changed data re-keys and retranslates.

Honest limits:

* This does not generalise. A V# assembled by `s_mov`s, one behind control flow,
  or one whose `s_load` has an unresolved soffset still refuses.
* `ShaderEmbeddedConstantLoads::LOADS_MAX` is 8, and these shaders already spend
  slots on their SRT loads. Overflow warns and drops the rest.
* Clearing this family will expose GTA V's next blocker. It was at 1.2 FPS with
  192 flips; nothing here makes it playable.
* The general fix is KytyPS5's shape — symbolic scalar provenance with
  memory-read nodes, resolvability as a graph query, bind-time evaluation
  through a guest-memory reader, and materialization into a real buffer binding.
  That needs plumbing this tree does not have: `scalar_eval` has no memory model,
  `ShaderEmbeddedConstantLoad` carries no producer/provenance, and
  `storage_buffers` can only be populated from the usage-slot table.

## Tests

* `raeen-hle`: `map_direct_memory_refuses_an_unplaceable_fixed_address_with_enomem`,
  `map_direct_memory_falls_back_to_the_physical_address_for_a_plain_hint`,
  `flexible_map_and_virtual_reserve_report_orbis_statuses_on_failure`.
* `raeen-runtime`: `every_measured_fixed_va_base_is_claimed_and_servable` —
  asserts over `TITLE_VA_WINDOWS` rather than one base, so adding a base cannot
  silently leave it undefended.
* `kyty-graphics`: `full_chain_vsharp_loaded_from_an_srt_table_to_validated_spirv`
  (real ISA bytes → parse → the production analysis entry point → recompile →
  spirv-val; verified to fail without the analysis change) and
  `vsharp_written_by_moves_is_still_refused`.
