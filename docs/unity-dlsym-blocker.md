# The Unity/IL2CPP `sceKernelDlsym` blocker

**Date:** 2026-07-28
**Branch:** `fix-dlsym-default-handle`
**Titles implicated:** Blasphemous II (measured), Subnautica: Below Zero
(same Unity/IL2CPP fingerprint, not independently measured)

---

## The measured symptom

A 50 s hardware run of Blasphemous II produced:

```
WARN raeen_hle::libkernel: sceKernelDlsym(handle=0, symbol='scriptingGetMem'):
       handle names NO registered module — ENOENT
...
frame path: reached=nothing | videoout_open=0 buffers_registered=0
            dcb_submitted=0 draws=0 flips_submitted=0 frames_published=0
STALL_DUMP IN-FLIGHT HLE:
  t1      = libScePosix::pthread_cond_wait
  t2..t15 = libkernel::sceKernelWaitSema      (14 threads, all parked)
  t15 top call = libkernel_unity::sceKernelRaiseException
```

The title never reaches VideoOut. IL2CPP asks libkernel for its scripting
allocator, is refused, raises, and the worker pool parks forever.

---

## Two defects, not one

The starting hypothesis was that handle 0 is POSIX `RTLD_DEFAULT` and that
`scriptingGetMem` is an IL2CPP-internal symbol in some module's export table.
**Both halves are wrong**, and the references say so unambiguously.

### 1. Handle 0 is the main program, not a global search scope

`RuntimeLinker::FindProgramById` — KytyPS5 `src/loader/runtimeLinker.cpp:1532`:

> `// Id 0 is reserved for main program`

It returns `m_programs.front()`. Program ids are handed out from 1
(`program->unique_id = ++id_seq`, L1271), so 0 can never collide with a real
module. Orbis defines **no** `RTLD_NEXT`-style sentinel in this API; there is
one reserved value and it names the executable.

Raeen's `OrbisKernel` already matches that shape — `next_module_id` starts at 1
and the executable registers first — so handle 0 maps to the lowest module id
carrying an export table. Our old code passed 0 through as an ordinary id,
matched nothing, and returned `ENOENT`.

### 2. `scriptingGetMem` is not a guest export and never will be

No amount of module-table searching finds it. It is an **allocator hook the
runtime is expected to supply**, and both references that boot Unity titles
special-case exactly this name:

| Reference | Location | Behaviour |
|---|---|---|
| KytyPS5 | `src/libs/libKernel.cpp:262` | `handle == 0 && symbol == "scriptingGetMem"` → returns `KernelApplicationHeapGetMem` |
| SharpEmu | `DirectExecutionBackend.Imports.cs:2098` | `TryResolveRuntimeSymbolAlias`: `"scriptingGetMem" => "malloc"`, plus `scriptingFreeMem`/`Realloc`/`Calloc` |

This is why the previous SharpEmu audit's "dlsym bootstrap argument
normalization" follow-up never closed the gap on its own: normalizing the
handle argument still leaves nothing to resolve *to*.

---

## Semantics adopted

Established from KytyPS5 first (it boots Blasphemous II in-game), SharpEmu
second. **shadPS4 has no real implementation** — `sceKernelDlsym` appears only
as an `aerolib.inl` name-table `STUB`, so it contributed nothing here.

`sceKernelDlsym(handle, symbol, addrOut)`:

1. `addrOut == 0` or an unreadable `symbol` → `SCE_KERNEL_ERROR_EFAULT`.
2. Resolve the scope: `handle == 0` → the main program; otherwise that handle.
3. Look the symbol's NID up in that module's export table.
4. **Fallback A** — every other loaded module, in load order (ascending
   handle). From SharpEmu's `DispatchKernelDynlibDlsym`, which tries the handle
   and then sweeps process-wide. Logged at WARN when it hits: a symbol found
   outside the module the guest named means our handle bookkeeping disagrees
   with the title's, which is worth knowing even though the call succeeded.
5. **Fallback B** — Raeen's own HLE trampolines, by name. Both references do
   the equivalent (Kyty returns emulator functions from `KernelDlsym`; SharpEmu
   calls them "runtime symbols"). A guest module export always wins over this.
6. Miss → `SCE_KERNEL_ERROR_ESRCH` (`0x80020003`), **not** `ENOENT`. KytyPS5
   returns `ESRCH` for both an unknown handle and an absent symbol. The
   out-pointer is left untouched and the miss is logged with handle, name, NID,
   the named module's export count, and the number of published HLE
   trampolines — three different failures needing three different fixes, which
   a bare error code cannot distinguish. That ambiguity is exactly how this bug
   survived several sessions being read as a memory fault.

### Why the load-order sweep is ordered

`lle_module_exports` is a `DashMap`; its iteration order is arbitrary. Two
modules may legally export the same NID. An unordered "first hit" would resolve
the same symbol to different addresses on different runs, so
`resolve_lle_export_anywhere` sorts handles ascending before searching.

---

## Does `scriptingGetMem` resolve now?

**Yes — through fallback B, not through any module export table.** It resolves
to a reserved HLE trampoline whose implementation is
`libkernel::scriptingGetMem`.

### The signature, and the guard on it

`scriptingGetMem(alignment, size) -> void*`, taken from KytyPS5's
`KernelApplicationHeapGetMem` (`src/libs/libKernel.cpp:203`), which clamps
`alignment` up to `0x10` and returns null for a non-power-of-two — a guard only
worth writing against observed arguments.

SharpEmu disagrees: it aliases the name to plain `malloc(size)`, which would
read the alignment as a size. **The reference that actually runs the title
wins.**

Raeen keeps the power-of-two check for the same reason KytyPS5 has it: it is a
**self-test on the signature**. If the real first argument were a size, it would
almost never be a power of two, so a mis-read ABI surfaces as a loud null return
instead of a plausible-looking pointer the guest writes `size` bytes through.

### What is deliberately *not* implemented

`scriptingRealloc` and `scriptingCalloc`. SharpEmu aliases them to libc
`realloc`/`calloc`, but if `scriptingGetMem` really is `(alignment, size)` then
this family does not use libc argument order, and guessing wrong on a *resize*
corrupts the guest heap. They fail with `ESRCH` and a log line naming the
symbol. KytyPS5 — which boots the title — implements only the `GetMem` half, so
there is no evidence the title needs them.

`scriptingFreeMem` **is** implemented: one pointer argument, no return, and
unambiguous whichever shape the rest of the family turns out to have. Handing
IL2CPP an allocator with no matching deallocator makes every scripting free
leak.

---

## The mechanism: trampoline reservation

`dlsym` can only hand back an address that already exists. Every other HLE
function earns its trampoline from a relocation — some module imported it, so
the linker minted an address and wrote it into the import slot. A function the
guest reaches *only* by name has no importer anywhere in the process.

Three pieces:

- `raeen_hle::libkernel::DLSYM_RESERVED_EXPORTS` names the dlsym-only exports.
- `ProcessTables::reserve_hle_export` (new, `raeen-firmware`) mints a
  trampoline with no relocation behind it, sharing one index space with the
  import-driven entries. `load_process` calls it for each reserved export.
- `publish_hle_exports_for_dlsym` (new, `raeen-runtime`) publishes the whole
  process trampoline table to the kernel keyed by function name, before guest
  entry, from all three process entry points.

A side effect worth noting: a title that `dlsym`s any *other* name Raeen
implements and the process imports (`malloc`, `pthread_create`, …) now resolves
it too, which it previously could not.

### The reservations must not be counted as imports

Caught by an existing `raeen-gui` test rather than by inspection: the Shell
reported `linked.hle_trampolines.len()` as "N HLE imports resolved", so the two
reservations became **two phantom resolved imports on every title**. Fixed with
`LinkedModule::imported_hle_trampoline_count()` /
`reserved_hle_trampoline_count()`, which exclude the reserved names.

Deliberately derived by name rather than stored as a serialized count: a
linked-process **cache hit** returns without ever calling `load_process`
(`raeen-firmware/src/process_cache.rs`), so a new field would have had to
round-trip correctly or every cached launch would misreport. The cache key
already hashes `lib.rs`, `dynlib/linker.rs`, and `hle.registered_names()`, all
three of which this change touches, so existing cache entries invalidate
cleanly. A round-trip test now asserts a restored process still carries the
reservation and still excludes it from the import count.

---

## `libkernel_unity`: already complete, no change

KytyPS5 registers exactly three functions under that library name
(`AddLibkernelUnityFunc`, `src/libs/libKernel.cpp:2998`, call sites L3081-3088):

| NID | Function | Registered by Raeen? |
|---|---|---|
| `Qhv5ARAoOEc` = `0x421bf90110283847` | `sceKernelRemoveExceptionHandler` | yes |
| `WkwEd3N7w0Y` = `0x5a4c0477737bc346` | `sceKernelInstallExceptionHandler` | yes |
| `il03nluKfMk` = `0x8a5d379e5b8a7cc9` | `sceKernelRaiseException` | yes |

All three are already aliased under both `libkernel` and `libkernel_unity`
(`crates/raeen-hle/src/libkernel.rs`, the `for library in ["libkernel",
"libkernel_unity"]` loop). The NIDs were verified against Raeen's own
`nid_names.txt` and re-encoded to confirm they match Kyty's strings exactly.

The trace's `libkernel_unity::sceKernelRaiseException` is corroborating rather
than contradicting evidence: it appears as an **in-flight HLE call**, which only
happens for a resolved export. `sceKernelDlsym` itself is imported from
`libkernel` — the log line proves it, since the message is emitted from inside
our handler. No aliasing work is needed.

---

## Honest limits

- **No retail title was re-run.** This agent cannot run retail titles. Every
  claim above is a semantics, mechanism, and test claim. That Blasphemous II
  now presents frames is a *reasoned prediction*, not a measurement.
- Resolving `scriptingGetMem` unblocks the dlsym call. Whether IL2CPP then
  reaches VideoOut depends on what it does next, which is unmeasured. The
  14-thread semaphore park is downstream of the raise; if the raise stops, the
  park should too, but that is inference.
- The allocator is backed by Raeen's guest heap, not by an application-heap API
  the title registered. KytyPS5 routes through `sceKernelRtldSetApplicationHeapAPI`
  (`api[6]`, `posix_memalign`) and returns null when the title has not
  registered one. Raeen always answers from its own heap, which is more
  forgiving and less faithful; revisit if a title's allocator bookkeeping turns
  out to care.

---

## Command to confirm on hardware

```powershell
cargo run --release -p raeen-gui -- --run-eboot "Games\Blasphemous II\eboot.bin"
```

(or launch the title from the Shell, which spawns the same `--run-eboot` child).

Look for, in order:

1. `sceKernelDlsym(handle=0, symbol='scriptingGetMem') -> 0x4000... (Raeen HLE
   trampoline)` at DEBUG — the fix firing.
2. No `scriptingGetMem` WARN and no `libkernel_unity::sceKernelRaiseException`
   in the stall dump.
3. `frame path:` with non-zero `videoout_open` / `flips_submitted`.

If `scriptingGetMem` resolves but `scriptingRealloc` or `scriptingCalloc` now
appears as a fresh `ESRCH` WARN, that is the measurement needed to implement
them — capture the argument registers before choosing a signature.

To confirm the `libkernel_unity` finding against the title's own import table
rather than against KytyPS5:

```powershell
cargo run --release -p raeen-gui -- --imports "Games\Blasphemous II\eboot.bin"
```

Expect `sceKernelDlsym` under `libkernel`, and nothing under `libkernel_unity`
beyond the three exception-handler functions.
