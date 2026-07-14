# Third-Party Notices

XPS5X is licensed under **GPL-2.0-only** (see [`LICENSE`](LICENSE)). It
incorporates ideas and re-implemented code derived from the third-party
projects listed below. XPS5X ships **no Sony code, keys, or firmware**; every
subsystem here is original Rust, written clean-room with respect to Sony's
proprietary SDK/headers, using only the community sources credited below.

---

## Kyty — PS4/PS5 emulator (reference & porting source)

- **Upstream:** https://github.com/InoriRus/Kyty
- **License:** MIT
- **Copyright:** © 2021 InoriRus
- **How XPS5X uses it:** Portions of XPS5X's GPU (GNM command-buffer parsing,
  PSSL → SPIR-V shader translation, RDNA/GCN → Vulkan) and HLE kernel/library
  layers are **re-implemented in idiomatic Rust with reference to Kyty's C++
  source.** Kyty's tree is cloned locally into the git-ignored `reference/`
  directory for study only; it is **never vendored, compiled, or committed**
  into XPS5X. The MIT license permits use, modification, and redistribution of
  such derived work provided the copyright notice below is retained — which
  this file does.

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

- **Upstream:** https://github.com/par274/sharpemu
- **License:** GPL-2.0
- **Copyright:** © SharpEmu authors
- **How XPS5X uses it:** A second-opinion reference alongside Kyty for PS5
  module loading (eboot/PRX/sysmodule chains), kernel-surface structure
  (Fiber, AMPR, PlayGo), and VideoOut/AGC bring-up. Patterns and behavior are
  **re-implemented in idiomatic Rust with reference to SharpEmu's C# source**;
  no C# is transliterated or vendored. SharpEmu's tree is cloned locally into
  the git-ignored `reference/` directory for study only; it is **never
  vendored, compiled, or committed** into XPS5X.

GPL-2.0 is the same license as XPS5X (GPL-2.0-only), so derived
re-implementations are license-compatible; this notice preserves attribution.

---

## Compiled Rust crate dependencies

Unlike the clean-room reference sources above (studied but never linked),
these crates.io dependencies are compiled into XPS5X (or its test binaries).
Only licenses compatible with GPL-2.0-only are used.

- **rspirv** — https://github.com/gfx-rs/rspirv — dual MIT / Apache-2.0, used
  here under its **MIT** option (Apache-2.0 is *not* GPLv2-linking-compatible;
  MIT is). **Test-only** dev-dependency of `xps5x-gpu`: it structurally
  validates the shader emitter's SPIR-V output in unit tests and is **not
  linked into the distributed emulator binary**.

---

## Not incorporated (ecosystem references only)

The following projects were evaluated. Their **code is not used** in XPS5X —
they are GPL-3.0, which is incompatible with this project's GPL-2.0-only
license, and they target real (jailbroken) PS5 hardware rather than emulation.
They are noted only as ecosystem references (e.g. for the homebrew payload
format and the set of `sceKernel*` calls real homebrew invokes):

- **cy33hc/ps5-payload-loader** — GPL-3.0 — on-console homebrew payload loader.
- **phantomptr/ps5upload** — GPL-3.0 — desktop → console file-transfer tool.

If any of their code were ever to be incorporated, XPS5X would first have to
move to GPL-3.0(-or-later); that has not been done.
