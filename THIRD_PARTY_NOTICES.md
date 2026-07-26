# Third-Party Notices

Raeen is licensed under **GPL-2.0-only** (see [`LICENSE`](LICENSE)). It
incorporates ideas and re-implemented code derived from the third-party
projects listed below. Raeen ships **no Sony code, keys, or firmware**; every
subsystem here is original Rust, written clean-room with respect to Sony's
proprietary SDK/headers, using only the community sources credited below.

---

## Kyty — PS4/PS5 emulator (reference & porting source)

- **Upstream:** https://github.com/InoriRus/Kyty
- **License:** MIT
- **Copyright:** © 2021 InoriRus
- **How Raeen uses it:** Portions of Raeen's GPU (GNM command-buffer parsing,
  PSSL → SPIR-V shader translation, RDNA/GCN → Vulkan) and HLE kernel/library
  layers are **re-implemented in idiomatic Rust with reference to Kyty's C++
  source.** Kyty's tree is cloned locally into the git-ignored `reference/`
  directory for study only; it is **never vendored, compiled, or committed**
  into Raeen. The MIT license permits use, modification, and redistribution of
  such derived work provided the copyright notice below is retained — which
  this file does.
- **Directly ported data:** the AGC Gen5 register-default tables in
  `crates/raeen-hle/src/libsce_agc_reg_defaults.rs` (served by
  `sceAgcGetRegisterDefaults2[Internal]`) are a faithful port of Kyty's
  `Graphics.cpp` `g_cx/sh/uc_reg_info1/2` tables with register names resolved
  against Kyty's `Pm4.h`.
- **Behavioral mapping:** `raeen-gpu`'s Gen5 stencil conversion follows Kyty's
  explicit AMD-operation mapping rather than treating AMD and Vulkan enum
  values as layout-compatible. Raeen's Rust implementation additionally
  validates unsupported operations and preserves the guest's separate test and
  operation reference values.

MIT is compatible with GPL-2.0: MIT-derived portions may be combined into this
GPL-2.0-only work, and this notice preserves the required MIT attribution for
those portions.

```
MIT License

Copyright (c) 2021 InoriRus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## KytyPS5 — active PS5 Kyty fork (reference & porting source)

- **Upstream:** https://github.com/Nmzik/KytyPS5
- **License:** GPL-2.0 with Kyty/MIT lineage
- **How Raeen uses it:** PS5-specific GPU, VM, pthread, and HLE behavior is
  compared selectively rather than merged wholesale. Raeen's runtime-owned
  pthread allocation adds a separate 1 MiB reserve after observing Minecraft
  enter a fixed 0x14a778-byte stack frame on a thread whose guest-visible
  attribute remains 1 MiB. This is an idiomatic Rust re-implementation of
  KytyPS5's separate `PTHREAD_STACK_EXTRA` behavior. Raeen also follows
  KytyPS5's Gen5 vertex-attribute handling by carrying the AGC `fetch_index`
  selector into Vulkan's per-vertex/per-instance input rate; the Rust data
  model, cache keys, and tests are original. No C++ is vendored or compiled
  into Raeen.

---

## SharpEmu — PS5 emulator (reference & porting source)

- **Upstream:** https://github.com/sharpemu/sharpemu (formerly par274/sharpemu)
- **License:** GPL-2.0-or-later (compatible with Raeen's GPL-2.0-only)
- **Copyright:** © SharpEmu authors
- **Reference synced:** 2026-07-26 to upstream `main` @ `0535783` (the
  `v0.0.2-beta.5` work is included) — brings working AudioOut2 audio (GTA V),
  AGC cross-queue
  `WAIT_REG_MEM` label work (`Agc/GpuWaitRegistry.cs`), and PR #587's Gen5 flat
  (global) memory + 3D-texture shader support (`Shader/*`, PSSL SEG-field
  FLAT-address decode → SPIR-V global-memory access). Ports cite the
  originating SharpEmu `file:line` in doc comments.
- **How Raeen uses it:** A second-opinion reference alongside Kyty for PS5
  module loading (eboot/PRX/sysmodule chains), kernel-surface structure
  (Fiber, AMPR, PlayGo), VideoOut/AGC bring-up, and **native host controller
  input** (XInput + raw-HID DualSense — `Host/Windows/WindowsXInputReader.cs`,
  `WindowsDualSenseReader.cs`, `WindowsHidNative.cs`, `Host/HostGamepadState.cs`,
  re-implemented as `crates/raeen-input/src/{xinput,hid,native}.rs`). The
  APR/AMPR async file-I/O path is also a SharpEmu port:
  `Ampr/AmprExports.cs` (`AprCommandBufferReadFile` eager read,
  `TryReadFileToGuestMemory` positional read-exact loop + host-handle cache,
  `CompleteCommandBuffer` record walk) and
  `Kernel/KernelAprCompatExports.cs` (submit/wait), re-implemented in
  `crates/raeen-hle/src/libsce_ampr.rs` and the
  `apr_complete_command_buffer` in `crates/raeen-hle/src/libkernel.rs`.
  The Gen5 AudioOut2 context/port parameter layout and grain pacing in
  `crates/raeen-hle/src/libsce_audio_out2.rs`, plus GFX10 MIMG DIM/NSA field
  meanings used by `crates/kyty-graphics/src/shader/parse.rs`, were also
  behaviorally re-implemented after comparison with SharpEmu.
  Type-11 guest cubemaps are lowered to six-layer Vulkan 2D-array views after
  comparing SharpEmu/KytyPS5's `(s,t,face)` image path and Raeen's measured
  Minecraft shader; the Rust implementation and regression tests are original.
  UserService's
  retail-style primary-user id and one-shot login-event behavior were
  re-implemented in `crates/raeen-hle/src/libsce_user_service.rs`; the event
  ABI was independently cross-checked against KytyPS5 and shadPS4. Raeen's
  resource-class-local descriptor indexing, layered guest tiling/writeback,
  and Vulkan staging-size guard are original Rust fixes derived from its own
  Minecraft traces and Vulkan validation output.
  Raeen's executable leaf-import gateway was also designed after auditing
  SharpEmu's native import trampoline state/ABI preservation; Raeen's
  implementation is original Rust plus generated x86-64 code and retains the
  existing VEH route for context-changing calls.
  From upstream `5228335` (PR #587, "Gen5 flat memory and 3D images"):
  `OpImageQuerySizeLod`'s result vector is sized from the descriptor's
  dimensionality (`%v3int` for 3D and 2D-array, `%v2int` for 2D) in
  `crates/kyty-graphics/src/shader/{spirv,recompile}.rs`, after comparing
  SharpEmu's `Gen5SpirvTranslator.cs`; the Rust emitter and its regression
  tests are original. From upstream `a158960` (PR #592, "GPU compute detile"):
  the row-parallel split of the CPU detile loop in
  `crates/raeen-gpu/src/texture/tiling.rs` follows SharpEmu's `Parallel.For`
  over the same loop in `Agc/GnmTiling.cs`, and the non-power-of-two
  element-size guard mirrors its `BitLog2` refusal. Raeen's swizzle-equation
  tables were verified independently equivalent to SharpEmu's for the modes
  both implement (5/9/24/27) and were **not** changed by that comparison.
  Patterns and behavior are **re-implemented in idiomatic Rust with reference to
  SharpEmu's C# source**; no C# is transliterated or vendored. SharpEmu's tree
  is cloned locally into the git-ignored `reference/` directory for study only;
  it is **never vendored, compiled, or committed** into Raeen.
- **NID name catalog:** SharpEmu's `scripts/ps5_names.txt` (a public symbol-name
  list) is used as candidate input to `merge_nid_catalog`, which admits a name
  only if Raeen's own SCE-NID hash reproduces the NID from it. The result is
  factual hash data (public symbol names, no Sony code/keys), folded into
  `crates/raeen-firmware/src/dynlib/nid_names.txt`.

GPL-2.0 is the same license as Raeen (GPL-2.0-only), so derived
re-implementations are license-compatible; this notice preserves attribution.

---

## shadPS4 — PS4 emulator (reference source; NID→name data incorporated)

- **Upstream:** https://github.com/shadps4-emu/shadPS4
- **License:** GPL-2.0-or-later (`SPDX-License-Identifier: GPL-2.0-or-later`;
  the repository's `LICENSE` is the GNU GPL **Version 2, June 1991** text)
- **Copyright:** © 2024 shadPS4 Emulator Project and contributors
- **How Raeen uses it:** Primarily an Orbis HLE reference (memory, libkernel,
  linker, Vulkan), re-implemented in Rust rather than transliterated.
  The 2026-07-26 refresh to `d976c33` exposed the stale-wake failure class in a
  condition-wide generation counter: a signal intended for one waiter can be
  observed by every waiter. Raeen's FIFO/per-waiter Rust condition queue and
  tests are an original implementation informed by shadPS4 commit `26f4270`;
  no C++ is copied.

  **Data incorporated in-tree:** `crates/raeen-firmware/src/dynlib/nid_names.txt`
  is derived from shadPS4's `src/core/aerolib/aerolib.inl` — a generated table
  of public SCE symbol names and their NIDs. Raeen uses it strictly as a
  **candidate dictionary**: an entry is admitted only if Raeen's own
  `dynlib::nid::nid_of()` reproduces the NID from the name, so every retained
  name is a verified SHA-1 preimage rather than a trusted assertion (94,247 of
  aerolib's 94,276 entries pass; 29 are rejected). The test
  `nid_names::tests::all_names_hash_to_their_nid` re-proves the entire table on
  every run. Regenerate with the adjacent `gen_nid_names.py`.

  These are **public symbol names, not Sony code** — no SDK headers, firmware,
  keys, or binaries are involved, consistent with `.claude/skills/clean-room`
  ("NID names from community databases OK"). shadPS4's tree itself is cloned
  only into the git-ignored `reference/` directory and is never compiled or
  committed.

GPL-2.0-or-later may be exercised under GPL-2.0 terms, so the incorporated data
is license-compatible with Raeen's GPL-2.0-only; this notice preserves
attribution as that license requires.

---

## ps5-payload-dev/sdk — PS5 payload SDK (NID candidate source, names only)

- **Upstream:** https://github.com/ps5-payload-dev/sdk
- **License:** GPL-3.0-only (repo-wide; `include/freebsd` files are BSD) —
  **incompatible for code**: nothing from this project is compiled, linked, or
  vendored into Raeen. Its tree is cloned locally into the git-ignored
  `reference/ps5-payload-sdk` directory only.
- **How Raeen uses it:** symbol *identifiers* from its public headers were used
  as **candidates** for the NID dictionary, via `merge_nid_catalog`. A
  candidate is admitted only when Raeen's own `dynlib::nid::nid_of()`
  reproduces a real NID from it — so what lands in
  `crates/raeen-firmware/src/dynlib/nid_names.txt` is factual hash data (a
  short functional identifier plus its independently recomputed SHA-1
  preimage), not copied SDK content. Measured 2026-07-25: 35,181 of 37,345
  candidates added new hash-verified names. This is the same admission rule
  `.agents/skills/clean-room` grants for community NID databases; no SDK code,
  headers, or build files were incorporated.

## idc/ps4libdoc — PS4 library documentation (consulted; nothing incorporated)

- **Upstream:** https://github.com/idc/ps4libdoc
- **License:** none stated in the repository.
- **Measured result:** its 42,010-name `known_names.txt` was run through the
  same hash-gated merge on 2026-07-25 and added **zero** new names — the
  existing shadPS4/SharpEmu-derived catalog already covered every entry.
  Nothing from this source is incorporated; it is recorded here because it was
  evaluated as a candidate source.

## Mesa AddrLib — AMD surface-layout reference (acquired; no code incorporated yet)

- **Upstream:** https://gitlab.freedesktop.org/mesa/mesa
- **Pinned reference:** `main` at `780727e68adc`
  in git-ignored `reference/mesa`.
- **License:** the acquired `src/amd/addrlib/` files carry
  `SPDX-License-Identifier: MIT` and AMD copyright notices. The reference's
  `licenses/MIT` text is retained in the local clone.
- **How Raeen uses it:** Phase 0 establishes this as the authoritative,
  machine-pinned source for later clean-room AddrLib tiling work. No Mesa code
  or tables have been copied into Raeen in this phase. Any later transcription
  must cite the exact source file/revision and preserve its MIT attribution.

---

## Compiled Rust crate dependencies

Unlike the clean-room reference sources above (studied but never linked),
these crates.io dependencies are compiled into Raeen (or its test binaries).
Only licenses compatible with GPL-2.0-only are used.

- **iced-x86** — https://github.com/icedland/iced — MIT, used by the module
  linker to identify real x86-64 `syscall` instructions in executable guest
  segments. Those instructions are trapped into the Orbis syscall dispatcher
  so a PS5 syscall number can never be executed against the Windows kernel.

- **rspirv** — https://github.com/gfx-rs/rspirv — dual MIT / Apache-2.0, used
  here under its **MIT** option (Apache-2.0 is *not* GPLv2-linking-compatible;
  MIT is). **Test-only** dev-dependency of `raeen-gpu`: it structurally
  validates the shader emitter's SPIR-V output in unit tests and is **not
  linked into the distributed emulator binary**.

- **naga** — https://github.com/gfx-rs/wgpu (naga crate) — dual MIT /
  Apache-2.0, used here under its **MIT** option. **Test-only** dev-dependency
  of `kyty-graphics` (`spv-in` feature): its SPIR-V front end parses the
  binaries produced by the `spirv_asm` assembler in unit tests as an extra
  validity gate. It is **not linked into the distributed emulator binary**
  through this use (naga also ships transitively inside the GUI's wgpu stack,
  which is an unrelated, already-present dependency).

---

## Not incorporated (ecosystem references only)

The following projects were evaluated. Their **code is not used** in Raeen —
they are GPL-3.0, which is incompatible with this project's GPL-2.0-only
license, and they target real (jailbroken) PS5 hardware rather than emulation.
They are noted only as ecosystem references (e.g. for the homebrew payload
format and the set of `sceKernel*` calls real homebrew invokes):

- **cy33hc/ps5-payload-loader** — GPL-3.0 — on-console homebrew payload loader.
- **phantomptr/ps5upload** — GPL-3.0 — desktop → console file-transfer tool.

If any of their code were ever to be incorporated, Raeen would first have to
move to GPL-3.0(-or-later); that has not been done.
