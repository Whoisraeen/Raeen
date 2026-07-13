# XPS5X Guest Address Space — Design Spec

**Date:** 2026-07-13
**Status:** Design (pending plan)
**Scope:** The runtime-owned **guest address space** ("arena") that gives executing guest code a real heap, `mmap` regions, and (later) a stack at addresses it can dereference **natively** — replacing the flat, image-only `MappedImage` view with one contiguous host region where *guest address == host address*. This is the step from "a linked module's code runs and can touch the image" (RT1/dispatch-context) to "guest code can `malloc`, use the returned pointer, and `mmap`."
**Builds on:** the RT0/RT1 runtime (`execute_linked`, VEH dispatch, `MappedImage`), the HLE dispatch context (`HleContext { kernel, mem }`, `GuestMemory`), and `xps5x_kernel::VirtualMemoryManager`.

---

## 1. The problem

Native execution means guest code runs on the host CPU as `extern "sysv64"` machine code. The CPU dereferences whatever numeric value a guest pointer holds **as a host address** — there is no translation layer between a guest pointer and the load/store it performs. Therefore, for any guest memory the code actually uses, **guest address must equal host address.**

Two memory models exist today and neither satisfies this for a real program:

- **Runtime `MappedImage`** — real host memory the guest executes from, but spans only the flat module image. Its `GuestMemory` impl treats guest vaddr `V` as *offset* `V` into an OS-chosen mapping; this has held only because RT0/RT1 test code is position-independent and never dereferences an absolute image pointer.
- **Kernel `VirtualMemoryManager`** — tracks emulated addresses (e.g. `next_anon_addr = 0x2000_0000_0000`) with **detached `Vec<u8>` backing**. Good bookkeeping, but the guest CPU cannot reach a `Vec`, so an address `mmap`/`malloc` returns from here **faults when the guest dereferences it.**

Consequence: `sceKernelMapFlexibleMemory`/`malloc` currently return sentinel or VMM addresses the guest cannot use. To run anything non-trivial we need one address space where allocations are real, dereferenceable host memory.

## 2. The decision: a fixed-base, identity-mapped guest arena owned by the runtime

The runtime reserves **one contiguous host virtual region at a fixed base**, and defines the guest address space to be **identity-mapped onto it**: guest address `A` *is* host address `A`. Everything the guest touches — image, heap, stack, `mmap` — lives inside this arena at its real address, so every guest pointer is a valid host pointer.

- **Fixed base, not OS-chosen.** Because guest pointers baked into the image by the linker's relocations must equal real host addresses, the arena base cannot be chosen by the OS at map time. It is a fixed constant, and the module is **linked at that base**. A high, normally-free address makes a fixed `VirtualAlloc(MEM_RESERVE)` reliable; failure is surfaced as `MapFailed`, not hidden.
- **The runtime owns host memory; the kernel VMM becomes metadata.** Only the runtime reserves/commits/frees host arena pages. `VirtualMemoryManager` stops owning `Vec` backing for arena regions and instead **records region metadata** (base, size, protection) so `is_mapped`/`region_containing`/`mprotect` keep working — its bytes are the arena's bytes. (Its existing `Vec`-backed path stays for any non-arena/legacy use and its own unit tests; arena regions use the new metadata path.)
- **Single active execution.** The whole model relies on the existing RT invariant that exactly one `execute_linked` runs at a time (`dispatch::CALL_LOCK`). The arena is reserved at the top of `execute_linked` and released at the end, so no two arenas ever coexist and the fixed base never races.

This is the standard native-ISA-emulator approach (a reserved guest window with identity mapping). Lazy per-page commit via a fault handler, W^X per-segment protection, and a 48-bit-scale window are later refinements (§7); this spec commits the whole (modest) arena up front for simplicity.

## 3. Arena layout

Fixed base `GUEST_ARENA_BASE = 0x0000_1000_0000_0000` (16 TiB — high, normally free, clear of the trampoline guard at `0x4000_0000_0000` and unresolved-stub sentinel at `0x5000_0000_0000`). Total reserved span `0x1_0000_0000` (4 GiB). Fixed sub-region offsets from the base:

| Region | Offset range (from base) | Size | Protection | Purpose |
|--------|--------------------------|------|------------|---------|
| Image  | `0x0` .. `0x4000_0000`   | 1 GiB | `RWX` (RT-shortcut) | the linked module image, mapped at `GUEST_ARENA_BASE` |
| Heap   | `0x4000_0000` .. `0x8000_0000` | 1 GiB | `RW` | `malloc`/`calloc`/`realloc`/`memalign` |
| Stack  | `0x8000_0000` .. `0xA000_0000` | 512 MiB | `RW` | guest stack (top = base+`0xA000_0000`, grows down) — reserved now, RSP-switch is §7 |
| Mmap   | `0xA000_0000` .. `0x1_0000_0000` | 1.5 GiB | `RW` | anonymous `mmap` allocations (bump up) |

**Link-base contract:** the module image must be linked so guest vaddr `0` lands at `GUEST_ARENA_BASE`. The runtime exports `pub const GUEST_ARENA_BASE`; the shell's `FirmwareLauncher` passes it to `load_module` as the load base (replacing `DEFAULT_LOAD_BASE = 0x8000_0000`), and `execute_linked` documents that `module.base` must equal `GUEST_ARENA_BASE`. Image larger than the 1 GiB image region → `MapFailed`.

## 4. Interfaces

New allocation seam in `xps5x-hle` (parallel to the existing `GuestMemory` trait, so `xps5x-hle` needs no dependency on `xps5x-runtime`):

```rust
/// Allocation within the guest address space, for HLE heap/mmap functions.
/// Addresses returned are real guest (== host) addresses the caller may then
/// read/write through `GuestMemory`. All return `None` on exhaustion / bad
/// request rather than panicking.
pub trait GuestAllocator {
    fn alloc(&self, size: u64, align: u64) -> Option<u64>;       // heap
    fn free(&self, addr: u64);
    fn realloc(&self, addr: u64, new_size: u64) -> Option<u64>;
    fn mmap(&self, length: u64, align: u64) -> Option<u64>;      // mmap region
    fn munmap(&self, addr: u64, length: u64);
}

pub struct HleContext<'a> {
    pub kernel: &'a xps5x_kernel::OrbisKernel,
    pub mem: &'a dyn GuestMemory,
    pub alloc: &'a dyn GuestAllocator,   // NEW
}
```

`HleFunction = fn(&HleContext, &[u64]) -> u64` is unchanged in shape; only `HleContext` grows a field, threaded exactly as `mem` was in the dispatch-context milestone.

## 5. Components

```
crates/xps5x-runtime/src/
  arena.rs   # NEW: GuestArena — fixed-base reserve/commit/free; heap + mmap
             #      bump/free-list allocators; impls GuestMemory + GuestAllocator
  mem.rs     # MappedImage retired in favour of arena.rs (or kept only for the
             #      image copy helper); GuestMemory now lives on GuestArena
  dispatch.rs# ActiveContext gains `alloc: *const dyn GuestAllocator`; veh_callback
             #      builds HleContext { kernel, mem, alloc }
  lib.rs     # execute_linked builds a GuestArena, maps image at GUEST_ARENA_BASE,
             #      passes &arena as both mem and alloc
```

- **`GuestArena`** — reserves `[GUEST_ARENA_BASE, +4 GiB)` (`VirtualAlloc(MEM_RESERVE)` at the fixed base), commits the image region `RWX` and heap/stack/mmap regions `RW`, copies the image in at offset 0, and frees the whole region (`VirtualFree(MEM_RELEASE)`) on `Drop`. Implements:
  - `GuestMemory` — identity: `read`/`write` bounds-check `[addr, addr+len)` against the committed regions (via `checked_add`, returning `false` on any out-of-range or overflowing request — no panic, no OOB) and copy to/from host address `addr` directly.
  - `GuestAllocator` — `alloc`: 16-byte-aligned bump pointer within the heap region plus a first-fit free list of returned blocks; `free`: push block to the free list; `realloc`: alloc-copy-free (bounded by the smaller size); `mmap`: page-aligned bump within the mmap region; `munmap`: best-effort (RT2b may no-op reclaim). Allocator state is interior-mutable (`Mutex`), sound under the single-active-execution invariant but self-contained.
- **`OrbisKernel`/`VirtualMemoryManager`** — gains a metadata-record method, e.g. `record_mapping(addr, size, prot)` inserting a `MemoryRegion` **without** `Vec` backing, so `is_mapped`/`region_containing`/`mprotect` see arena mappings. Existing `Vec`-backed `mmap`/`read`/`write` remain for legacy/tests.

## 6. HLE wiring

- `libc::malloc(size)` → `ctx.alloc.alloc(size, 16)`; `calloc(n, sz)` → alloc then zero via `ctx.mem.write`; `realloc(p, n)` → `ctx.alloc.realloc`; `free(p)` → `ctx.alloc.free`; `memalign/posix_memalign(align, size)` → `ctx.alloc.alloc(size, align)`. All return `0`/`NULL` on `None` (honest OOM), never panic. Replaces the `FAKE_HEAP_ADDR` sentinel.
- `libkernel::sceKernelMapFlexibleMemory`/`sceKernelAllocateDirectMemory`/`mmap` → `ctx.alloc.mmap(len, PAGE)`, then `ctx.kernel.memory.record_mapping(addr, len, prot)`, write the out-param address through `ctx.mem`, return `SCE_OK`/the address. `munmap` → `ctx.alloc.munmap` + VMM record removal.
- `ctx.mem` reads/writes now span the whole committed arena, so the existing `memcpy`/`memset`/`strlen` real-behavior work automatically covers heap and mmap addresses, not just the image.

## 7. Non-goals (later milestones)

- **Dedicated guest stack / RSP switch, TLS/`fsbase`.** The stack region is reserved and committed, but RT2 still calls the guest on the *host* thread's stack (works for the code run so far). Switching RSP into the arena stack before the call (a small inline-asm trampoline) and setting up TLS is a later milestone.
- **Lazy per-page commit** (commit-on-fault in the VEH) and a **larger / 48-bit-scale** window — RT2 commits a fixed 4 GiB up front.
- **W^X per-segment protection** — the image region stays `RWX` (the documented RT shortcut).
- **`VirtualAlloc2` placeholder reservation** for robust fixed-base claiming — RT2 uses plain fixed `MEM_RESERVE` and surfaces failure as `MapFailed`.
- Real file-backed `mmap`, `mprotect` enforcement on host pages, `munmap` page decommit.

## 8. Milestones

- **RT2a — Arena + heap.** `GuestArena` reserves the fixed-base region, maps the image at `GUEST_ARENA_BASE`, and implements `GuestMemory` (identity) + `GuestAllocator` (heap). `execute_linked` uses it; the launcher/linker link at `GUEST_ARENA_BASE`. `malloc`/`calloc`/`realloc`/`free`/`memalign` route to it. **Acceptance:** a guest entry calls `malloc(N)`, writes a pattern into the block, reads it back (all through the real HLE + arena) equal to what was written; the returned address is inside the heap region (outside the image); `free` then `malloc` reuses. Existing RT execute tests pass (updated for identity addressing at `GUEST_ARENA_BASE`).
- **RT2b — `mmap` into the arena + VMM metadata.** `mmap`/`munmap` route to the arena's mmap region and record/remove `VirtualMemoryManager` metadata; `is_mapped`/`region_containing` reflect arena mappings. **Acceptance:** a guest `mmap(len)` returns an mmap-region address, the guest reads/writes it, and `kernel.memory.is_mapped(addr)` is true; `munmap` removes the record.
- **RT2c+ (future, §7):** RSP switch + guest stack, TLS/`fsbase`, lazy commit, W^X, POSIX backend.

## 9. Verification

- **RT2a automated (`cargo test`, Windows):** a synthetic linked module (built through the real LM1 linker at `GUEST_ARENA_BASE`) whose entry: calls `malloc(0x40)` via its HLE trampoline, `memset`s the returned block, reads byte 0 back into `RAX`; `execute_linked` returns the written value and the block address is asserted within `[GUEST_ARENA_BASE + heap_off, ...)`. Plus unit tests on `GuestArena`: alloc alignment, free-list reuse, `GuestMemory` bounds rejection (wild address → `false`, no panic), arena reserve/free is leak-clean across repeated `execute_linked` calls.
- **RT2b automated:** `mmap` roundtrip + `kernel.memory.is_mapped` true; `munmap` clears it.
- **Guardrail:** crate stays `#![forbid(unsafe_op_in_unsafe_fn)]`; every `unsafe` (fixed-base reserve, per-region commit, identity read/write, whole-arena free) carries a `SAFETY:` note; clippy clean; no real firmware bytes; no keys.

## 10. Global constraints

- Rust edition 2024, rust-version ≥ 1.85, GPL-2.0-only. No new external dependencies (`windows-sys` already covers `VirtualAlloc`/`VirtualFree`; `GuestAllocator` and arena allocator are hand-written).
- Windows-first; the `arena`/`mem`/`dispatch` mechanism stays `#[cfg(target_os = "windows")]`, the public `execute_linked` signature unchanged and platform-independent (the non-Windows stub still returns `MapFailed`).
- Clean-room and trust boundary unchanged (runtime design §6): the runtime executes only images the LM1 pipeline produced from inputs the user supplied; no keys, no firmware, no circumvention.
- Bounds-checking is total: every `GuestMemory`/`GuestAllocator` entry point returns a sentinel (`false`/`None`/`0`) on an out-of-range, overflowing, or exhausted request — never a panic, never an OOB host access.
