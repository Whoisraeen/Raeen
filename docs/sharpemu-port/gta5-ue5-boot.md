# SharpEmu port: GTA V / UE5 boot unblockers

Reviewed 2026-07-28 against the SharpEmu tree at its live tip (`0535783f`;
the mission brief's `92e3abe` is an ancestor). Baseline Raeen state:
`docs/gta5-blocker-analysis-2026-07-27.md` — GTA V links every NID, presents
4 frames, then dies on a guest stack-canary smash on thread 31; Until Dawn
(UE5) exits at ~6.7 s the same way.

Verdicts are per commit, with the root cause and Raeen's equivalent.

---

## 1. `a1cbff8` (#454) — NID `BHouLQzh0X0` + doubled static TLS reservation

### 1a. NID `BHouLQzh0X0` — **ALREADY-HAVE**

`BHouLQzh0X0` is `sceKernelDirectMemoryQuery`. Raeen implements it in
`crates/raeen-hle/src/libkernel.rs` (registered at ~:1908, implementation at
~:3016), derived from shadPS4's `memory.cpp`, with tests. SharpEmu's
`KernelMemoryCompatExports.KernelDirectMemoryQuery` has the same shape
(find the allocated region owning the offset, fill
`SceKernelDirectMemoryQueryInfo`, `ENOENT`/`EACCES` when nothing owns it).
No change needed.

### 1b. Static TLS reservation — **N/A (structurally different, and Raeen is not undersized)**

SharpEmu reserves a **fixed** Variant-II prefix below the thread pointer:

```
// src/SharpEmu.HLE/GuestTlsTemplate.cs:20
public const ulong StartupStaticTlsReservation = 0x20000UL;  // Was 0x10000UL, but thats too small for GTA V
```

`RegisterModule` hard-fails when the cumulative static offset exceeds it, so a
fixed reservation *can* be undersized — hence their 64 KiB → 128 KiB doubling.

**Raeen has no such constant and no such cap.** It sizes the area from the
actual module set at launch:

- `crates/raeen-firmware/src/sprx.rs:145` — `static_tls_total(layout)`
- `crates/raeen-runtime/src/arena.rs` `setup_main_tcb` — allocates
  `static_tls_total(tls_layout) + TCB_SIZE` (`TCB_SIZE = 0x800`) and places the
  TCB at `base + area`, with each module's `.tdata` at `area - tp_offset`.
- `crates/raeen-runtime/src/thread.rs:607` — a worker calls `setup_thread_tcb`
  with the same layout, so every thread gets an identically-sized area.

So the answer to "how much does Raeen reserve, and is it enough for GTA V" is:
**exactly what the linked module set asks for, so it cannot be too small for
the modules known at launch** — there is no before/after number to change. The
doubling is not portable and would be a regression to a fixed cap.

**Residual risk, unfixed and worth measuring** (not the canary bug, and out of
scope for this pass): the area is sized once from the launch-time
`module.tls_layout`. SharpEmu additionally models rtld's *lazy* DTV for modules
that appear **after** a thread was seeded (`CreateDynamicEntry` allocates a
fresh block instead of assuming space below a live thread pointer). If a Raeen
title `sceKernelLoadStartModule`s a module carrying `PT_TLS` after threads
exist, that module has no static slot. Raeen's `__tls_get_addr`
(`crates/raeen-hle/src/libkernel.rs:3311`, over
`raeen_kernel::static_tls_area_offsets`) should be audited for what it returns
for an unknown module id before any title is blamed on it.

---

## 2. `db4339f` (#650) — restore wiped GTA foundation (PPSA04264)

**Verdict: mostly ALREADY-HAVE / N/A.** The current SharpEmu tree has **no
per-title compatibility database and no title-ID branching**; everything that
commit restored lives in title-agnostic paths. Only two places name GTA V:

1. The static-TLS reservation above (§1b) — N/A for Raeen.
2. One uncatalogued `libKernel` NID captured from the title,
   `Ikfdt-rIqCE` (`src/SharpEmu.Libs/Stubs/GameServiceStubs.cs:115`),
   registered as a **side-effect-free success** stub under the synthetic name
   `sceUnknownIkfdt`, with the comment refusing to guess the ABI ("reverse the
   ABI before writing guest memory"). Raeen measures **zero unresolved NIDs**
   on this title, so this is not a live blocker; the honest reason not to add
   it blind is that an explicit "return 0" for an unreversed function is
   behaviourally identical to Raeen's existing loud-unresolved handling, minus
   the diagnostic. Noted, not ported.

The AudioOut2 out-buffer overruns from that batch were already fixed in Raeen
(`7c220d8`) and were explicitly out of scope here.

---

## 3. `2764aaa` (#542) — ctype tables / data-object imports → `_Ctype`

**Verdict: PORTED (the `_Ctype` data export).**

Root cause class: some imports are **data**, not functions. An HLE registry can
only hand out a trampoline (a *code* marker); a data symbol needs a real
readable guest address. `_Ctype` was one of only two symbols still unresolved
across all measured titles, and an unresolved data import means the guest
dereferences a marker address on its first `isalpha`/`printf` directive.

SharpEmu has **no `_Ctype` export**, but it has the mechanism and the exact
table convention:

- `HleDataSymbols.cs` allocates a plain zeroed host block per data symbol and
  hands its address to the guest; `MergeKnownHleDataSymbols` injects them only
  when a real guest definition is absent.
- The *same* NID can be registered both as a data object and as a callable
  export (`__stack_chk_guard` is, returning the same object address) — the
  dual-registration pattern.
- `LibcStdioExports.cs:24-26,37-45,736-738`: 384 `u16` entries covering
  `-128..=255`, and *"the pointer handed to the guest must point at the c == 0
  entry, not the start of the allocation."* Flag bits `_XD 0x001, _UP 0x002,
  _SP 0x004, _PU 0x008, _LO 0x010, _DI 0x020, _CN 0x040, _BB 0x080, _XB 0x400`.

Raeen already had that identical convention and flag layout in
`crates/raeen-hle/src/libc.rs` behind the `_Getpctype()` **function**. What was
missing was the data spelling.

Implemented:

- `crates/raeen-hle/src/libc.rs` — new `pub fn ctype_class_table_bytes()` and
  `pub const CTYPE_TABLE_ZERO_SLOT_OFFSET` (256). `write_ctype_table` (the
  `_Getpctype` path) now serializes through the same helper, so the data and
  function spellings **cannot drift apart**. A title that resolves `_Ctype` in
  one translation unit and calls `_Getpctype()` in another must not classify
  the same character two different ways.
- `crates/raeen-firmware/src/lib.rs` — `build_hle_data_page` embeds the table
  and registers `_Ctype` at `table_offset + 256` (the `c == 0` slot), under
  **both** provider views a title may name (`libc`, `libSceLibcInternal`) since
  Raeen's `resolve` is provider-aware. Added to
  `hle_data_page_export_names()`.

Test: `hle_data_page_exports_ctype_table_at_the_zero_slot` pins the NID
(`0x7ae9_7630_2de0_698b`, the `nid_names.txt` identity), resolution under both
providers, byte-for-byte equality with `_Getpctype`'s table, and the indexing
contract (`'A'`→`0x003`, `'z'`→`0x010`, `'5'`→`0x021`, `-1`→`0`).

**`_Ctype` is now resolvable.** Caveat: whether a title indexes `_Ctype[c]`
(matching this c==0-slot convention) or biases the index itself is not
observable from Raeen's own tree; the convention chosen is the same one
`_Getpctype` already commits to and the one SharpEmu documents.

---

## 4. `daaeb62` (#406) UE5 boot + `90c72eb` (#451) UE mutex

### 4a. `daaeb62` — **PORTED. This is the highest-value UE5 finding.**

Root cause, in SharpEmu's own words
(`KernelMemoryCompatExports.cs:5090-5096`):

> Unreal Engine titles depend on this: their base directory is
> `<app>/binaries/<platform>`, so they address content with `"../../../"`
> prefixes that land back inside /app0 on real hardware. Combining those raw
> against the app0 root walked out of the game folder entirely, so the title
> enumerated an unrelated host directory and never found its .pak files.

Raeen had **the opposite behaviour**: `combine_within_mount`
(`crates/raeen-kernel/src/filesystem/mod.rs`) *refused* any path containing a
`..` segment, and `open()` separately rejected any writable open of such a
path. Raeen had already ported SharpEmu's *later* sandbox hardening (`e01092a`)
but took refusal where SharpEmu normalizes. For a UE5 title (Until Dawn) that
denies every content and `Saved/` path the engine emits.

Implemented — `..` now **pops** the last resolved segment and is **silently
dropped at the mount root** (the clamp), and the redundant blanket rejection in
`open()` is gone. This is not a weakening: popping can only shorten the segment
list, so the by-construction lexical containment is unchanged, and the
drive-qualifier (`:`)/absolute-segment refusal, the reparse-point walk, and the
canonical-containment assertion all still run on the normalized result.

Tests:
- `unreal_project_relative_paths_resolve_back_into_app0` — the real UE shape
  `/app0/binaries/prospero/../../../Project/Content/Paks/pakchunk0.pak`
  resolves and **reads the actual file**, and the same shape works for a
  writable `Saved/` open that landed inside the mount.
- `parent_traversal_tail_is_clamped_to_the_mount_root` — replaces
  `parent_traversal_tail_is_denied`; asserts clamping *and* containment,
  including `/app0/../../../../../../escape.bin`.
- `writable_open_of_traversing_path_is_confined_not_refused` — replaces the old
  refusal test; asserts the write lands inside the mount and that nothing is
  created above the root.
- `mount_root_and_children_resolve_without_prefix_collisions` updated for the
  new contract.

Not ported from the same family (deliberate): SharpEmu's AvPlayer
`file://../../../` URI unwrapping (`AvPlayerExports.cs:1198-1341`) — same
root cause in the media path. Worth a follow-up if a UE title's video fails.

### 4b. `90c72eb` — **N/A here: OVERLAPS the concurrent mutex agent. Not edited.**

Per instructions I did not touch `crates/raeen-hle/src/pthread_sync.rs`. The UE
mutex behaviours in SharpEmu's `KernelPthreadCompatExports.cs` are exactly that
agent's territory, so they are reported rather than implemented:

- **Default mutex type is ERRORCHECK, not NORMAL** (`NormalizeMutexType`:
  `0 → 1`; a NULL `attr` yields ERRORCHECK). This is a real divergence from
  Linux/glibc that a Rust port could easily get wrong.
- **`MutexTypeAdaptiveNp` (4) self-relock is idempotent**, not counted:
  *"Gen5 runtime wrappers can layer an adaptive lock call over
  `scePthreadMutexLock` for one logical acquisition, followed by only one
  unlock."* Guarded by `IsGuestTrackedSelfLock` (guest qword at `mutex+8`
  already naming this thread ⇒ genuine deadlock ⇒ `EDEADLK`).
- **NORMAL (3) self-relock recurses instead of returning `EDEADLK`**:
  *"Several Gen5 runtimes layer their own owner/count bookkeeping over a NORMAL
  kernel mutex. Returning EDEADLK here leaves that guest bookkeeping out of
  sync … and turns the wrapper into a permanent lock/unlock retry loop."*
- **`trylock` may barge past queued waiters** (POSIX gives it no fairness
  obligation); gating it on an empty wait queue lets one stale waiter wedge a
  spin-on-trylock loop forever while the mutex is free.
- **Stale-waiter pruning**: one thread may have at most one pending
  acquisition; a leftover entry (typically a `cond_timedwait` timeout whose
  re-acquire hand-off was lost) clogs the FIFO head so the unlock hand-off
  wakes a dead wake-key.
- Static initializer sentinel: `*mutex == 1` means a statically-initialized
  **adaptive** mutex; `*mutex == 0` lazily creates an ERRORCHECK one.
- Direct hand-off on unlock is the #439 work already in flight.

Type constants (Orbis numbering): `ERRORCHECK 1, RECURSIVE 2, NORMAL 3,
ADAPTIVE_NP 4`; object layout `type @ handle+0x20`, `protocol @ handle+0x3C`;
attr `type @ +0x00`, `protocol @ +0x04`.

---

## 5. `d7bd814` (#565) dlsym normalization + `336286e` (#414) TLS pattern scan

### 5a. `d7bd814` — **NOT PORTED this pass; concrete follow-up recorded.**

SharpEmu's `sceKernelDlsym`
(`DirectExecutionBackend.Imports.cs:2024-2061`), ABI
`(handle=rdi, name=rsi, out=rdx)` → RAX `0` / `-1`:

1. The handle is **truncated to `int32`** — the only normalization; there is no
   special case for `0`, `-1`, `-2`, or a module name passed as a handle.
2. Name read as ASCIIZ capped at **512 bytes**.
3. Four-stage resolution, first hit wins: per-module table by name then by
   computed NID → **global table by name, ignoring the handle**, with `_`-prefix
   fuzzing (exact, strip one leading `_`, prepend `_`) → global table by Sony
   NID → hardcoded aliases (`scriptingGetMem→malloc`, `…FreeMem→free`,
   `…Realloc→realloc`, `…Calloc→calloc`).
   Because stages 2-4 ignore the handle, **an invalid or foreign handle still
   resolves** — that is the effective "normalization": the handle is a hint.
4. `out == 0`, or a failed store, discards the resolution and returns `-1`.

The **bootstrap** case (`DispatchBootstrapBridge:2149-2181`) is a synthetic
payload whose first argument is an *operation/table pointer*, not a module
handle; it delegates verbatim to the dlsym dispatcher and relies on the
handle-ignoring global stages to succeed. Sibling
`il2cpp_api_lookup_symbol` (NID `r8mvOaWdi28`) is a **2-argument** form
(`rdi=name`, `rsi=out`) that additionally zeroes `*out` on failure.

Follow-up for Raeen: audit `sceKernelDlsym` for (a) int32 handle truncation,
(b) a handle-ignoring global fallback, (c) `_`-prefix fuzzing, (d) NID
fallback, (e) not writing `*out` on failure.

### 5b. `336286e` — **N/A with reason.**

That commit fixes an **off-by-one in a code scanner**: `FindTlsAccessPatterns`
changed its bound to `ptr <= start + length - pattern.Length` so a match ending
at the last byte of the scanned buffer is found (`JitStubs.cs:249-264`). The
scanner exists because SharpEmu **rewrites guest instructions** — it patches
`mov reg, fs:[0]` into a call to a TLS handler, because it cannot give the
guest a real FS base (notably under Rosetta).

Raeen does not patch guest code for TLS: it installs a real guest TCB via
`RDFSBASE`/`WRFSBASE` and re-arms the FS base after Windows preemption
(`crates/raeen-runtime/src/tls.rs`, `dispatch.rs`). There is no scanner to fix.

**Explicitly do not port** SharpEmu's fourth matcher
(`TryPatchStackCanaryInstruction`, `DirectExecutionBackend.cs:3242-3306`),
which rewrites `mov/xor reg, fs:[0x28]` into `xor reg, reg` so the canary always
reads zero. That is a Rosetta workaround and is precisely the
zero-canary soft-success the `m1-homebrew` skill names as an anti-pattern.

---

## 6. `0c467e8` (#450) — add missing NIDs — **ALREADY-HAVE (registration), with a caveat**

Raeen measures zero unresolved NIDs on both titles, so there is nothing to add.
The portable insight is about **resolution semantics**, not the list:

SharpEmu keys dispatch on the **NID alone** — `LibraryName` on
`SysAbiExport` is descriptive metadata that never enters any lookup key, and a
compile-time analyzer (`SHEM001`) makes a duplicate NID an error. Across all
1071 declarations, **no export name appears under more than one library**, and
`LibraryName` defaults to `"libKernel"` when omitted.

Raeen resolves per `(provider, NID)`. Consequence when reading SharpEmu as a
source of truth: **its library names are hints and may not match what a given
title's `PT_SCE_DYNLIBDATA` actually imports the symbol from** — several are
`libKernel` only by default. This is the same trap already documented in
`build_hle_data_page` (registering the IPv6 constants only under `libkernel`
left Minecraft's `libSceNet` import unresolved), and it is why `_Ctype` above is
registered under both `libc` and `libSceLibcInternal`.

SharpEmu also supports one name under two NIDs (title-captured aliases, e.g.
two NIDs for `sceUserServiceGetUserName`) — a pattern Raeen may need if a title
imports an aliased NID.

---

## What this pass changed

| Area | File | Change |
|---|---|---|
| Stack guard | `crates/raeen-firmware/src/lib.rs` | `stack_chk_guard()` — one process-wide canary via `OnceLock`, now `pub` |
| Stack guard | `crates/raeen-runtime/src/arena.rs` | every TCB's `fs:0x28` uses that same word |
| UE5 paths | `crates/raeen-kernel/src/filesystem/mod.rs` | `..` normalized with a mount-root clamp; redundant `open()` refusal removed |
| `_Ctype` | `crates/raeen-hle/src/libc.rs` | shared table generator + zero-slot offset, public |
| `_Ctype` | `crates/raeen-firmware/src/lib.rs` | `_Ctype` data export under `libc` + `libSceLibcInternal` |

### The canary fix, stated plainly

This is the item most likely to move the measured blocker, and it came out of
§1's investigation rather than from the commit's literal content.

Raeen had **N + 1 independent random guard words**: `raeen-firmware` minted one
for the `__stack_chk_guard` global, and `raeen-runtime`'s `stack_canary()` was
called afresh inside `setup_main_tcb` for *every* TCB — so the main thread,
each of the ~30+ workers, and the global all differed. The old code documented
this as safe on the theory that "compiled code reads the same one in both
prologue and epilogue, so the two values need not agree."

Two real title behaviours break that theory, and both end in
`__stack_chk_fail` on a stack that was never corrupted:

1. **Mixed guards in one image** — a title links objects/middleware built with
   `-mstack-protector-guard=global` alongside others using the default `tls`
   (`fs:0x28`). Inlining across that boundary puts a prologue that loaded the
   global in the same frame as an epilogue comparing `fs:0x28`.
2. **A frame that spans threads** — GTA V's job system and UE5's task graph
   hand a suspended frame to a different worker. A canary spilled on one thread
   and verified on another can never compare equal against a per-TCB random
   value.

Real hardware has no such divergence: Orbis libkernel picks one guard word and
copies *that* word into each thread's TCB. SharpEmu reaches the same conclusion
independently — `HleDataSymbols.cs`: *"Keep the process data symbol and every
per-thread TLS copy byte-for-byte identical."*

Pinned by `every_tcb_and_the_global_share_one_stack_chk_guard`
(`arena.rs`) and `hle_data_page_publishes_the_one_process_wide_stack_chk_guard`
(`raeen-firmware`), both of which assert cross-thread and global equality **and**
keep the nonzero + NUL-terminator-byte properties so this cannot regress into a
zero-canary soft success.

**Honest status:** this is a *tested* fix for a real divergence on the exact
failing path, not a measured fix for the blocker. It has not been re-run
against the retail titles. Whether the thread-31 smash was caused by guard
divergence or by genuine memory corruption elsewhere is only decidable by
re-measuring.
