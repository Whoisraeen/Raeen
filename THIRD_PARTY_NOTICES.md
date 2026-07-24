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

## SharpEmu — PS5 emulator (reference & porting source)

- **Upstream:** https://github.com/sharpemu/sharpemu (formerly par274/sharpemu)
- **License:** GPL-2.0-or-later (compatible with Raeen's GPL-2.0-only)
- **Copyright:** © SharpEmu authors
- **Reference synced:** 2026-07-23 to upstream `main` @ 6db095e (tag
  `v0.0.2-beta.5`) — brings working AudioOut2 audio (GTA V), AGC cross-queue
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
  behaviorally re-implemented after comparison with SharpEmu. UserService's
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
