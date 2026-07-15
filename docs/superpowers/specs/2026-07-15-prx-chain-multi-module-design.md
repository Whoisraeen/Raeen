# M1-D: file-backed `.prx` chain + multi-module address space

**Date:** 2026-07-15
**Status:** Design, ready to implement. Every fact below is **measured** against a
real retail PS5 title, not assumed.

## Why this is the single highest-leverage task in the project

Measured on a real title (`xps5x --load-sprx <eboot>`, ranking added in `e871f49`):

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
$ xps5x --load-sprx Games/Minecraft/PPSA17221-app/libfmod.prx
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

## The one real gap: a multi-module address space

`GuestArena::new(&module.image)` maps exactly **one** image at
`GUEST_ARENA_BASE`. A title needs its eboot *and* its dependencies resident
simultaneously, each at a distinct base, with exports registered at
`base + vaddr`.

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

`xps5x --run-eboot Games/Minecraft/PPSA17221-app/eboot.bin` reports
`unresolved` dropping from 87222 to a few hundred (the remaining HLE gap), and
the guest advances past the `0x5000_0000_0000` (`UNRESOLVED_STUB_ADDR`) fault it
dies at today. That fault address moving is the honest signal of progress — it is
how every blocker this session was found.
