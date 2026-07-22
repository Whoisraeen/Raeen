# M1-D: file-backed `.prx` chain + multi-module address space

**Date:** 2026-07-15
**Status:** Design, ready to implement. Every fact below is **measured** against a
real retail PS5 title, not assumed.

## Why this is the single highest-leverage task in the project

Measured on a real title (`raeen --load-sprx <eboot>`, ranking added in `e871f49`):

```
unresolved imports by library (most-wanted first):
   86852  libfmod              <-- 99.6% of ALL unresolved
      54  libRenoirCore.PS5
      52  libSceRtc
      21  libSceNpWebApi2
      17  libSceAcm
      16  libSceIme
      12  libSceSaveData_native
       9  libSceNpManager  ... (single digits each)
     101  <library unknown>
```

**86852 of 87222 unresolved imports (99.6%) are `libfmod`** — a third-party audio
engine that **ships with the title** (`libfmod.prx` sits next to `eboot.bin`).
It can never be HLE'd; it is the game's own code and must be *loaded*.

So this one feature removes ~99.6% of the unresolved set. The genuine HLE gap
behind it is small (tens per library, ~200 functions total) — not the thousands
the raw count implies. **Do this before any M2 work**: nothing can render while
the title cannot finish linking.

## What is already proven working (do not redo)

`libfmod.prx` **already loads** with the current loader — same SELF container,
same dynamic model, nothing new needed to read it:

```
$ raeen --load-sprx Games/Minecraft/PPSA17221-app/libfmod.prx
SELF reassembled (real layout): 0 decrypted, 6 passed through, 14 phdr(s), 1752372 ELF byte(s)
Parsing .sprx module: e_type=0xfe18          <- ET_SCE_DYNAMIC (a shared library)
module uses the standard vaddr-based dynamic model (51 tag(s) mapped)
imports: 148  exports: 1112                  <- 1112 EXPORTS
```

Those 1112 exports are exactly what the eboot's 86852 `libfmod` relocations
reference. The pieces to consume them already exist:

* `ModuleRegistry::register_module_exports(name, &exports)` — already called for
  the main module in `load_module`.
* `Resolver::Lle { addr }` — `link_module` already handles it and writes
  `addr + addend` into the slot. **Resolution is NID-keyed and module-agnostic**,
  so once a dependency's exports are registered, the main module's imports find
  them with no further work.
* `link_module(module, dynlib, registry, hle, base)` already takes a **base**.

## The one real gap — and it is SMALLER than it looks

`GuestArena::new(&module.image)` maps exactly **one** image at
`GUEST_ARENA_BASE`. A title needs its eboot *and* its dependencies resident
simultaneously, each at a distinct base, with exports registered at
`base + vaddr`.

**Do NOT rewrite `GuestArena`.** Measured facts that collapse this:

* The arena's image region is **1 GiB** (`IMAGE_OFFSET = 0x0`,
  `IMAGE_SIZE = 0x4000_0000`, `arena.rs:39-40`).
* A real eboot's image spans ~`0xf49b720` (~256 MB) of that.
* Its four bundled `.prx` are small — `libfmod.prx` reassembles to 1.75 MB;
  `libcohtml`/`libRenoirCore`/`MediaDecoders` are comparable. Low tens of MB
  total.

So **every module fits in the existing image region with ~768 MB to spare**, and
the whole problem reduces to *composing one flat image*:

1. Lay out: eboot at image offset 0; each dependency at a page-aligned offset
   above it (bump allocator over reassembled image sizes).
2. `link_module(dep, ..., base = GUEST_ARENA_BASE + dep_offset)` — `link_module`
   **already takes a base**.
3. Register each dep's exports at its absolute address.
4. Concatenate the linked images into one buffer (deps' bytes at their offsets).
5. `GuestArena::new(&combined)` — **unchanged**, it just sees one bigger image.

`load_module` currently returns a `LinkedModule` (which owns its `image`), so
composition belongs in a new `load_process`-style entry point above it; the
existing single-module path stays for fixtures/diagnostics.

### Sketch

1. **Resolve the dependency set.** For each `DT_NEEDED` (already decoded into
   `DynlibData::needed_modules`), look for that filename **next to the eboot**
   (`Games/<title>/<TITLEID>-app/libfmod.prx`). A `DT_NEEDED` with no file and no
   HLE library is the honest "cannot load" case — log it loudly (`load_module`
   already logs NEEDED coverage). Note the ones that ARE HLE-covered
   (`libc`/`libkernel`/`libSce*`) must NOT be file-loaded even if present.
2. **Assign bases.** Lay each module out at a distinct, page-aligned base inside
   the arena. The eboot's own image already spans ~0xf49b720 (~256 MB) of vaddr
   space starting at 0, so dependencies must go **above** it. The arena is 4 GiB
   at `GUEST_ARENA_BASE`, so there is room; a simple bump allocator over
   module image sizes (rounded up to a page) is sufficient. Record
   `(module_name -> base)`.
3. **Load dependencies first, main module last.** For each dependency:
   `decrypt_self -> parse_sprx -> standard_dynamic_view/parse_dynlibdata ->
   link_module(at its base) -> registry.register_module_exports(...)`. Exports
   must be registered with **absolute** addresses (`base + export.vaddr`) — check
   `SymbolExport`'s current semantics (module-relative today) and make the base
   explicit rather than implicit.
4. **Map every image into one arena.** This is the actual code change:
   `GuestArena` must accept a set of `(base, image)` rather than a single image.
   Keep the identity map (`host addr == guest addr`) — everything downstream
   (VEH, trampolines, `fs:` TLS) depends on it.
5. **Dependencies have their own imports too** (`libfmod` imports 148, of which
   67 already resolve to HLE and 85 do not). Resolve them the same way — the
   registry is shared, so ordering matters only in that a module's exports must
   be registered before whoever imports them links.

### Watch out for

* **`e_type=0xfe18` (ET_SCE_DYNAMIC)** vs the eboot's `0xfe10` — already accepted
  by `parse_sprx`; no change needed.
* **Non-zero load bias.** `link_module`'s doc admits it treats `e_entry` as an
  image offset and assumes `p_vaddr`s start at 0 ("a real `.sprx` with a non-zero
  load bias would need `entry - load_bias`"). Dependencies loaded at a non-zero
  base are exactly that case — this is the known `todo` row in the ledger and
  must be handled here.
* **Module init.** A real `.prx` expects its init (`DT_INIT`/`.init_array`) to run
  before use. Out of scope for first light, but `libfmod` will not function
  without it — expect a second round.
* Do **not** special-case any title. This is the generic PS5 layout; it must hold
  for the next game too.

## Acceptance

`raeen --run-eboot Games/Minecraft/PPSA17221-app/eboot.bin` reports
`unresolved` dropping from 87222 to a few hundred (the remaining HLE gap), and
the guest advances past the `0x5000_0000_0000` (`UNRESOLVED_STUB_ADDR`) fault it
dies at today. That fault address moving is the honest signal of progress — it is
how every blocker this session was found.

---

## POST-IMPLEMENTATION CORRECTION (chain built in `c7a764b`)

The chain is built and **works**: three real `.prx` load at distinct bases,
register absolute exports, and compose into one image (no `GuestArena` change,
as predicted). But the headline number above is **suspect**.

Measured directly against the files:

* eboot has **876** distinct import NID strings; only **129** appear anywhere in
  `libfmod.prx`.
* Loading the chain dropped `unresolved` 87222 -> **87106** (only 116) — i.e.
  ≈ the 129 that genuinely match. The chain resolved what it legitimately could.
* 129 distinct symbols cannot account for 86852 unresolved *relocations* unless
  each were referenced ~673x; and if that were so, resolving 116 of them would
  have dropped `unresolved` by tens of thousands, not 116.

**Conclusion: the `library_index -> name` attribution in the by-library ranking
(`--load-sprx`) is probably WRONG** — a `DT_SCE_IMPORT_LIB` `id` (`val >> 48`)
may not share numbering with a symbol's `#lib#` field. So "99.6% is libfmod" is
likely an artifact, and the real distribution of the remaining ~87k unresolved
relocations is **unknown**.

### Do this first, before any more HLE or M2 work

1. **Verify the attribution.** Take one known-unresolved NID, find its symtab
   entry, decode its `#lib#`, map it through `import_libs`, and confirm that
   library actually exports that NID. If it doesn't, the mapping is wrong.
2. **Find what the 87k relocations actually are.** They may not be imports at
   all — count them by `r_type` and by whether `symbols[r_sym].is_import`. Note
   `linked.unresolved` holds one entry *per relocation*, so a handful of symbols
   can dominate the count.
3. Only then rank libraries and decide what to implement.

The honest signal remains the fault address (`--run-eboot`): it still stops at
`0x5000_0000_0000` (`UNRESOLVED_STUB_ADDR`). Move that, and you've made progress.

---

## RESOLVED — measured against the files (all three questions answered)

The suspicion above was correct. Everything below was measured with an
**independent** Python ELF parser (not our crate — that would be circular) and
then confirmed by the fixed tool reporting the same numbers.

### 1. The attribution was WRONG. The mechanism

A real PS5 title carries **two parallel `(id << 48) | (version << 32) |
name_off` tables**, and we indexed the wrong one:

| tag | table | ids | count (measured) |
|-----|-------|-----|------------------|
| `0x6100_0045` `DT_SCE_NEEDED_MODULE_1` | **modules** (`.prx` files) | 1..=50 | 50, 1:1 with `DT_NEEDED` |
| `0x6100_0049` `DT_SCE_IMPORT_LIB_1` | **libraries** | 0..=53 | 54 |

A module contains several libraries — module `libkernel` provides libraries
`libkernel`, `libScePosix`, `libSceCoredump`, `libkernel_write_throttling` — so
the tables have different lengths and different numbering (effectively off by
one across the shared prefix).

`dynlib/mod.rs` named `0x6100_0045` `DT_SCE_IMPORT_LIB` and looked a symbol's
`#lib#` id up in it. Library id 3 is `libc`; **module** id 3 is `libfmod`.
Hence "86852 = libfmod". The true answer is **`libc`** — the same 86852
relocations, a different library. Kyty's tag table
(`DT_OS_NEEDED_MODULE_1 = 0x61000045`, `DT_OS_IMPORT_LIB_1 = 0x61000049`,
MIT © InoriRus) independently confirms the naming.

Field order was *not* the bug: `<nid>#<library>#<module>` is correct, matching
Kyty's `Resolve` (`ids.At(1)` = library, `ids.At(2)` = module). Confirmed on a
real symbol: `...#q#X` = library `libScePosix` (42) in module `libkernel` (23).

Three independent confirmations, any one of which is decisive:

* **Name recovery.** The top symbol `pZ9WXcClPO8` (48292 refs) brute-forces to
  `_ZTVN10__cxxabiv120__si_class_type_infoE`; `byV+FWlAnB4` (23330) to
  `_ZTVN10__cxxabiv117__class_type_infoE`; `zr094EQ39Ww` (13566) to
  `__cxa_pure_virtual`. C++ RTTI vtables — `libc`, not an audio engine.
* **Export coverage.** `libfmod.prx` exports **0** of the 347 symbols the old
  mapping blamed on it. It exports exactly **54** of the eboot's NIDs — exactly
  what the corrected mapping attributes to `libfmod`.
* **The chain's own result.** Corrected: `libfmod` 54 + `libcohtml.Prospero` 62
  = **116** — precisely the 87222 -> 87106 drop `c7a764b` produced.

### 2. What the 87k relocations ARE

All **87414** symbol relocations target genuine `SHN_UNDEF` imports — 0 target
defined symbols, 0 have a bad NID, 0 are out of range. The "may not be imports
at all" worry is **refuted**. But `r_type` is everything:

| r_type | count | what it is |
|--------|-------|-----------|
| `R_X86_64_64` | 86592 | a **data pointer** slot (RTTI/vtable) — never called |
| `R_X86_64_JUMP_SLOT` | **758** | a function the guest **calls** |
| `R_X86_64_GLOB_DAT` | 64 | a data pointer slot |

Six C++ ABI symbols generate **86460** of them (98.9%): four `__cxxabiv1`
typeinfo vtables, `__cxa_pure_virtual`, and one more. Every polymorphic class
in a 254 MB C++ binary points its typeinfo at the same few vtables.

**So the real HLE gap is 570 unresolved called functions, not 87222.** The scary
number was never the work.

### 3. The corrected ranking (`--load-sprx`, post-fix)

```
   relocs    called  library
    86852       232  libc
       62        38  libcohtml.Prospero
       54        54  libfmod
       52        52  libScePosix
       27        26  libkernel
       21        21  libSceNpWebApi2
       17        17  libSceAgc
```

`libc` + `libScePosix` + `libkernel` = **310 of 570** called functions (54%),
all classic HLE territory. The bundled `.prx` supply 92 (libfmod 54 +
libcohtml 38). Loading `libfmod` was never the 99.6% lever — it is a 0.06% one.

### What this changes

* Rank HLE work off the **called** column, never the relocation count.
* `is_import = st_shndx == 0 || st_value == 0` is theoretically sloppy but
  **harmless here**: the eboot's dynsym is 877 entries, *all* `SHN_UNDEF` with
  `st_value == 0`, so both predicates agree exactly. Not a bug to chase.
* Fixed in `dynlib/mod.rs` (both tables decoded, `import_modules` added) and
  `linker.rs` (`UnresolvedImport { nid, r_type }`), pinned by
  `library_ids_index_the_library_table_not_the_needed_module_table`.

### Still true: the fault address is the honest signal

`--run-eboot` still stops at `0x5000_0000_0000`. **Every unresolved symbol
shares that one stub address**, so when the guest faults there we cannot say
which import it wanted. The next step is a per-NID unresolved stub —
`UNRESOLVED_STUB_BASE + i * 8`, exactly the pattern `HLE_TRAMPOLINE_BASE`
already uses — so the fault reports "guest called `<nid>` from `<library>`,
unimplemented". That turns one opaque address into a worklist drawn from the
570, in call order.
