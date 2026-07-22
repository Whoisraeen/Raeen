# Guest Address Space (RT2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give executing guest code a real, identity-mapped guest address space (arena) with a usable heap and `mmap`, so `malloc`/`mmap` return addresses the guest can dereference natively.

**Architecture:** The runtime reserves one fixed-base host region (`GUEST_ARENA_BASE = 0x0000_1000_0000_0000`, 4 GiB) and identity-maps the guest onto it (guest addr == host addr). The image maps at the base; heap/stack/mmap are fixed sub-regions. A new `GuestAllocator` trait (in `raeen-hle`, alongside `GuestMemory`) is implemented by the runtime's `GuestArena` and threaded into `HleContext` so `malloc`/`mmap` HLE functions allocate real guest memory. The kernel `VirtualMemoryManager` becomes region *metadata* over the arena.

**Tech Stack:** Rust 2024, `windows-sys` (`VirtualAlloc`/`VirtualFree`, already a dep), existing VEH runtime.

**Spec:** `docs/superpowers/specs/2026-07-13-raeen-guest-address-space-design.md` (read §2–§6 for the model; this plan implements milestones RT2a + RT2b).

## Global Constraints

- Rust edition 2024, rust-version ≥ 1.85, GPL-2.0-only. **No new external dependencies.**
- Windows-first: `arena`/`mem`/`dispatch` stay `#[cfg(target_os = "windows")]`; `execute_linked`'s public signature is unchanged and platform-independent; the non-Windows stub still returns `MapFailed`.
- `raeen-runtime` stays `#![forbid(unsafe_op_in_unsafe_fn)]`; **every** `unsafe` block carries a `SAFETY:` comment; clippy clean across `--workspace --all-targets`.
- **Total bounds-checking, no panics:** every `GuestMemory`/`GuestAllocator` entry point returns a sentinel (`false`/`None`) on an out-of-range, overflowing (`checked_add`), or exhausted request — never a panic, never an OOB host access.
- Clean-room / trust boundary unchanged: no keys, no firmware bytes (synthetic buffers only), the runtime executes only LM1-pipeline images.
- Arena layout constants (from base): image `0x0`, heap `0x4000_0000`, stack `0x8000_0000` (top `0xA000_0000`), mmap `0xA000_0000`, end `0x1_0000_0000`. Heap/mmap allocs 16-byte / page aligned respectively.
- `GUEST_ARENA_BASE` is the single source of truth for the link base: the runtime exports it; the shell links modules at it.
- Fixed-base reservation + the single-active-execution invariant (`dispatch::CALL_LOCK`) together mean **only one `GuestArena` may exist at a time.** Any test that constructs a real `GuestArena` must serialize via `dispatch::call_lock()` (or an equivalent shared lock) so parallel tests don't collide on the fixed base.

---

### Task 1: `GuestAllocator` seam through HLE + runtime

**Files:**
- Modify: `crates/raeen-hle/src/lib.rs` (add `GuestAllocator` trait; add `alloc` field to `HleContext`; thread through `HleRegistry::call`; add a `TestAllocator` double in tests)
- Modify: `crates/raeen-runtime/src/dispatch.rs` (`ActiveContext.alloc: *const dyn GuestAllocator`; `run` param; `veh_callback` builds `HleContext { kernel, mem, alloc }`)
- Modify: `crates/raeen-runtime/src/lib.rs` (temporary: pass a stub allocator so the workspace compiles and existing tests pass — replaced in Task 3)
- Modify: `crates/raeen-hle/src/{libkernel,libc}.rs` etc. only if `HleContext` construction sites need it (they take `&HleContext`, so no change beyond the registry)
- Test: `crates/raeen-hle/src/lib.rs` tests updated to pass a `TestAllocator`

**Interfaces:**
- Produces: `pub trait GuestAllocator { fn alloc(&self,size:u64,align:u64)->Option<u64>; fn free(&self,addr:u64); fn realloc(&self,addr:u64,new_size:u64)->Option<u64>; fn mmap(&self,length:u64,align:u64)->Option<u64>; fn munmap(&self,addr:u64,length:u64); }` and `HleContext { kernel, mem, alloc }`.
- Consumes: existing `HleContext { kernel, mem }`, `GuestMemory`, `HleRegistry::call(&self, ctx, lib, fn, args)`.

**Steps:**
- [ ] Add `GuestAllocator` trait and the `alloc: &'a dyn GuestAllocator` field to `HleContext` in `raeen-hle`. Update every `HleContext { .. }` construction and `HleRegistry::call` usage. In hle unit tests, add a minimal `TestAllocator` (e.g. a bump over a `Cell<u64>`, `free`/`munmap` no-ops) and pass it.
- [ ] In `dispatch.rs`, add `alloc: *const dyn GuestAllocator` to `ActiveContext` (same raw-pointer + lifetime-erasing-transmute discipline as `mem`, with the same `SAFETY:` reasoning). Add an `alloc: &dyn GuestAllocator` param to `run`; build `HleContext { kernel, mem, alloc }` in `veh_callback`.
- [ ] In `lib.rs`, define a temporary private `NullAllocator` (all methods return `None`/no-op) and pass `&NullAllocator` into `run` so the crate compiles. Add a `// TODO(Task 3): replaced by GuestArena` note.
- [ ] Run `cargo test -p raeen-hle -p raeen-runtime` and `cargo build --workspace`. Expected: all existing tests pass (behavior unchanged — nothing calls `alloc` yet).
- [ ] Commit: `feat: GuestAllocator seam through HleContext + runtime dispatch`.

**Acceptance:** workspace builds; all prior tests green; no HLE function behavior changed yet.

---

### Task 2: `GuestArena` (fixed-base reserve + heap/mmap allocators + GuestMemory)

**Files:**
- Create: `crates/raeen-runtime/src/arena.rs`
- Modify: `crates/raeen-runtime/src/lib.rs` (add `mod arena;`, export `pub const GUEST_ARENA_BASE`)
- Test: unit tests in `arena.rs`

**Interfaces:**
- Produces: `pub const GUEST_ARENA_BASE: u64 = 0x0000_1000_0000_0000;` and `pub(crate) struct GuestArena` with `fn new(image: &[u8]) -> Result<GuestArena, RuntimeError>` (reserves the region, commits sub-regions, copies `image` to the image region), `fn entry_ptr(&self, entry_offset: u64) -> Result<*const u8, RuntimeError>`, and `impl raeen_hle::GuestMemory` + `impl raeen_hle::GuestAllocator` for it.
- Consumes: `RuntimeError`, `windows-sys` `VirtualAlloc`/`VirtualFree`.

**Details:**
- `new`: `VirtualAlloc(GUEST_ARENA_BASE as *, 0x1_0000_0000, MEM_RESERVE, PAGE_NOACCESS)` at the fixed base; on null → `MapFailed`. Then `VirtualAlloc(MEM_COMMIT)` the image region (page-rounded `image.len()`, capped at the 1 GiB image region → `MapFailed` if larger) as `PAGE_EXECUTE_READWRITE`, and the heap+stack+mmap regions as `PAGE_READWRITE`. `copy_nonoverlapping` the image into the image region. Store base + committed bounds.
- `GuestMemory` (identity): `read`/`write` validate `guest_addr >= GUEST_ARENA_BASE`, `checked_add(len)` doesn't overflow, and `[addr, addr+len)` lies within a single committed region; then `copy_nonoverlapping` to/from host address `addr` (which equals the guest address). Any failure → `false`.
- `GuestAllocator`: `Mutex`-guarded state. `alloc(size, align)`: align-up a bump pointer within `[base+heap_off, base+stack_off)`, first-fit reuse from a free-list `Vec<(addr,size)>`; on heap exhaustion → `None`. `free`: push `(addr,size)` to the free list (size tracked in an `addr→size` map). `realloc`: `alloc(new)` + copy `min(old,new)` bytes (via raw copy within the arena) + `free(old)`; `None` if alloc fails. `mmap(len, align)`: page-align bump within `[base+mmap_off, base+end)`; `None` on exhaustion. `munmap`: best-effort record removal (may no-op reclaim in RT2a).
- `Drop`: `VirtualFree(base, 0, MEM_RELEASE)`.

**Steps:**
- [ ] Write `arena.rs` with `GuestArena::new`, `entry_ptr`, `Drop`, and the two trait impls, each `unsafe` block carrying a `SAFETY:` note.
- [ ] Unit tests (each real-`GuestArena` test acquires `dispatch::call_lock()` first — see Global Constraints): (a) `new` succeeds and `entry_ptr(0)` is `GUEST_ARENA_BASE`; (b) `alloc` returns a 16-aligned address inside the heap region, distinct across calls; (c) `free` then `alloc` reuses the freed block; (d) `mmap` returns a page-aligned address inside the mmap region; (e) `GuestMemory` write-then-read roundtrip inside an allocated block; (f) `read`/`write` of a wild/out-of-arena address returns `false` (no panic); (g) constructing, dropping, and re-constructing a `GuestArena` succeeds (fixed base is reusable → no leak).
- [ ] `cargo test -p raeen-runtime`, `cargo clippy -p raeen-runtime --all-targets`. Expected: pass, clean.
- [ ] Commit: `feat: GuestArena — fixed-base identity guest address space with heap + mmap`.

**Acceptance:** `GuestArena` unit tests pass on Windows; clippy clean; no panics on wild addresses; arena is leak-clean across construct/drop cycles.

---

### Task 3: Wire `execute_linked` to `GuestArena` + align link base

**Files:**
- Modify: `crates/raeen-runtime/src/lib.rs` (`execute_linked` builds a `GuestArena`, maps image at base, passes `&arena` as both `mem` and `alloc`; drop the `NullAllocator` stub and `MappedImage` usage)
- Modify: `crates/raeen-runtime/src/mem.rs` (retire `MappedImage`, or keep only an image-copy helper if `arena.rs` reuses it — no dead code)
- Modify: `crates/raeen-gui/src/launcher.rs` (`DEFAULT_LOAD_BASE` → `raeen_runtime::GUEST_ARENA_BASE`)
- Modify: `crates/raeen-runtime/tests/execute.rs` (existing tests: link synthetic modules at `GUEST_ARENA_BASE`; guest-memory addresses in the `memcpy` test become `GUEST_ARENA_BASE + offset`)

**Interfaces:**
- Consumes: `GuestArena` (Task 2), `GUEST_ARENA_BASE`.
- `execute_linked` signature unchanged; `entry_offset` is still an offset into `module.image` (host addr = `GUEST_ARENA_BASE + entry_offset`).

**Steps:**
- [ ] Replace the `MappedImage` + `NullAllocator` path in `execute_linked` with `GuestArena::new(&module.image)?`, `arena.entry_ptr(entry_offset)?`, and `dispatch::run(entry, .., &arena /*mem*/, &arena /*alloc*/, &guard)`. Document that `module.base` must equal `GUEST_ARENA_BASE`.
- [ ] Point `FirmwareLauncher` at `GUEST_ARENA_BASE` for the load base.
- [ ] Update `tests/execute.rs`: build synthetic modules through the real linker at `GUEST_ARENA_BASE`; update the `memcpy` test's guest addresses to arena-absolute; keep the unresolved-trampoline and genuine-fault tests meaningful (a wild fault address is still outside the arena/guard).
- [ ] `cargo test -p raeen-runtime -p raeen-gui`, `cargo clippy --workspace --all-targets`. Expected: green, clean.
- [ ] Commit: `feat: execute_linked runs on the GuestArena (identity-mapped at GUEST_ARENA_BASE)`.

**Acceptance:** existing RT execute tests pass against the arena; the shell links + runs at `GUEST_ARENA_BASE`; workspace clippy clean.

---

### Task 4: libc heap functions on the arena (RT2a acceptance)

**Files:**
- Modify: `crates/raeen-hle/src/libc.rs` (`malloc`/`calloc`/`realloc`/`free`/`memalign`/`posix_memalign` → `ctx.alloc` + `ctx.mem`; remove the `FAKE_HEAP_ADDR` sentinel)
- Test: `crates/raeen-hle/src/libc.rs` unit tests (with `TestAllocator` + `TestMemory`); `crates/raeen-runtime/tests/execute.rs` (end-to-end)

**Steps:**
- [ ] Implement `malloc(size)` → `ctx.alloc.alloc(size,16)` (→ `0` on `None`); `calloc(n,sz)` → checked `n*sz`, alloc, zero via `ctx.mem.write`; `realloc(p,n)` → `ctx.alloc.realloc`; `free(p)` → `ctx.alloc.free`; `memalign(align,size)`/`posix_memalign` → `ctx.alloc.alloc(size,align)`. All honest-OOM (`0`/`NULL`), no panics.
- [ ] Unit tests: `malloc` returns nonzero distinct addresses; `calloc` block reads back zero; `free`+`malloc` reuse; OOM path returns 0.
- [ ] **RT2a end-to-end** in `tests/execute.rs`: a synthetic linked module (real linker, base `GUEST_ARENA_BASE`) whose entry calls `malloc(0x40)` through its HLE trampoline, `memset`s the block to a known byte, reads byte 0 back into `RAX`, returns. Assert the return equals the written byte **and** (separately, via the arena/return value) the block address is within the heap region. This runs on Windows and proves the whole path.
- [ ] `cargo test -p raeen-hle -p raeen-runtime`, `cargo clippy --workspace --all-targets`. Expected: green, clean.
- [ ] Commit: `feat: libc heap (malloc family) allocates real guest memory via the arena`.

**Acceptance:** the RT2a end-to-end malloc→memset→read test genuinely runs on Windows and returns the written value; `malloc` no longer returns a sentinel.

---

### Task 5: `mmap`/`munmap` into the arena + VMM metadata (RT2b)

**Files:**
- Modify: `crates/raeen-kernel/src/memory/mod.rs` (add `record_mapping(&self, addr, size, prot)` inserting a `MemoryRegion` with no `Vec` backing; `remove_mapping(&self, addr)`)
- Modify: `crates/raeen-hle/src/libkernel.rs` (`sceKernelMapFlexibleMemory`/`sceKernelAllocateDirectMemory`/`mmap` → `ctx.alloc.mmap` + `ctx.kernel.memory.record_mapping` + write out-param via `ctx.mem`; `munmap` → `ctx.alloc.munmap` + `remove_mapping`)
- Test: `crates/raeen-kernel/src/memory/mod.rs` (record/remove metadata), `crates/raeen-runtime/tests/execute.rs` (end-to-end mmap)

**Steps:**
- [ ] Add `record_mapping`/`remove_mapping` to `VirtualMemoryManager` (metadata only — `regions` map, no `backing` entry). Keep existing `Vec`-backed `mmap`/`read`/`write` and their tests intact.
- [ ] Route the HLE mmap family to `ctx.alloc.mmap(len, PS5_PAGE_SIZE)` + `record_mapping`; write the returned address through `ctx.mem` to the out-param where the ABI requires; return `SCE_OK`/address. `munmap` mirrors.
- [ ] Unit test `record_mapping`: `is_mapped`/`region_containing` reflect a recorded arena region; `remove_mapping` clears it.
- [ ] **RT2b end-to-end** in `tests/execute.rs`: a guest entry calls `mmap(len)` via HLE, writes+reads the returned region (through the arena), returns a marker; separately assert (via the shared kernel) `kernel.memory.is_mapped(addr)` is true for the returned address. `munmap` then clears it.
- [ ] `cargo test --workspace`, `cargo clippy --workspace --all-targets`. Expected: all green, clean.
- [ ] Commit: `feat: mmap/munmap allocate arena memory and record VMM metadata`.

**Acceptance:** the RT2b mmap roundtrip runs on Windows; `kernel.memory.is_mapped` reflects arena mappings; full workspace test + clippy clean.

---

## Self-review notes

- **Spec coverage:** Task 1 = §4 interfaces; Task 2 = §5 `GuestArena` + §3 layout; Task 3 = §3 link-base contract + §5 `execute_linked`; Task 4 = §6 heap + §8 RT2a acceptance; Task 5 = §6 mmap + §5 VMM metadata + §8 RT2b. §7 non-goals (RSP switch, TLS, lazy commit, W^X) are explicitly out.
- **Type consistency:** `GuestAllocator` method set is identical in spec §4 and Task 1/2. `GUEST_ARENA_BASE` value identical in spec §3 and here. `HleContext { kernel, mem, alloc }` identical.
- **Ordering:** trait (1) → arena impls trait (2) → integration + base alignment (3) → heap HLE + RT2a proof (4) → mmap HLE + RT2b proof (5). Each task ends with an independently testable, committable deliverable.
