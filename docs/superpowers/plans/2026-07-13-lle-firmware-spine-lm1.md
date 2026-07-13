# LLE Firmware Spine — LM1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A homebrew / already-decrypted `.sprx` flows through the full spine — SELF → decrypt-or-passthrough → inner ELF → `PT_SCE_DYNLIBDATA` parse → NID link against the HLE registry → relocated module — and reaches a defined, inspectable linked state. **No keys required.**

**Architecture:** Build on the LM0 foundation (`Firmware`, `KeyProvider`/`require_key`, `nid_of`/`encode_nid`/`decode_nid`) in `crates/xps5x-firmware`. Add the SELF decryption machinery (driven by the user `KeyProvider`, with plaintext passthrough for homebrew), an `.sprx` module parser, the `PT_SCE_DYNLIBDATA` decoder, a NID↔name database seeded from HLE function names, a `ModuleRegistry` that dispatches each import to HLE or LLE, and a linker that applies SCE relocations.

**Tech Stack:** Rust 2024; `aes` + `cbc` (workspace deps) for segment decryption; `sha1` (already used by `nid.rs`); `goblin`/`scroll` for ELF primitives; `xps5x-loader` for SELF/ELF structs; `xps5x-hle::HleRegistry` as the HLE resolution target.

**Authority:** `docs/superpowers/specs/2026-07-12-xps5x-lle-firmware-spine-design.md` (§3 components, §5 slice, §6 verification). LM1 = roadmap row LM1.

## Global Constraints

- **Clean-room boundary (non-negotiable, design §2):** XPS5X ships no keys, no firmware, no key-extraction/circumvention tooling. The crypto layer *consumes* a user-supplied key; it never derives, guesses, brute-forces, or extracts one. **No real firmware bytes in any test — synthetic buffers only.**
- Missing-key decryption is a **normal, expected, non-fatal** condition: raise `FirmwareError::MissingKey { key_id }`, log at `info`.
- Rust edition 2024, rust-version ≥ 1.85, GPL-2.0-only. Parsers **never panic** on malformed input — bounds-checked, return `FirmwareError`. Guard every length/offset read from the file against the buffer (the LM0 SLB2 `file_count` over-alloc fix is the standard to follow).
- `cargo build/test/clippy --workspace` clean after every task. No new crate dependencies beyond the existing workspace set.
- New public error variants use the existing `FirmwareError` enum (`UnsupportedRelocation(u32)`, `MalformedDynlibData(String)`, `MissingKey`, etc.); extend it only if a genuinely new failure mode appears, and wire through `XPS5XError`.

---

### Task 1: SELF decrypt-or-passthrough (`crypto/self_crypto.rs`)

**Files:**
- Create: `crates/xps5x-firmware/src/crypto/self_crypto.rs`
- Modify: `crates/xps5x-firmware/src/crypto/mod.rs` (add `pub mod self_crypto;` + re-export)
- Test: in-module `#[cfg(test)]`

**Interfaces:**
- Consumes: `KeyProvider`, `KeyRequest`, `SegmentKey`, `require_key` (from `crypto/mod.rs`); `xps5x_loader::self_format::{SelfHeader, SelfEntry, parse_self}`.
- Produces:
  ```rust
  /// The plaintext inner ELF image recovered from a SELF, plus which
  /// segments were decrypted vs. passed through.
  pub struct DecryptedSelf {
      pub elf: Vec<u8>,          // the inner ELF ready for sprx.rs
      pub decrypted_segments: usize,
      pub passthrough_segments: usize,
  }

  /// Recover the inner ELF from a SELF image. Encrypted segments are routed
  /// through `provider`; plaintext/homebrew segments pass through unchanged.
  /// Returns `FirmwareError::MissingKey` if an encrypted segment needs a key
  /// the provider does not have.
  pub fn decrypt_self(data: &[u8], provider: &dyn KeyProvider)
      -> Result<DecryptedSelf, FirmwareError>;
  ```

**Approach:** Parse the SELF header/entry table (reuse `xps5x-loader` structs; do not re-hardcode). For each segment: if the entry's flags mark it plaintext (homebrew / already-decrypted — the LM1 path), copy it through; if it marks it encrypted, build a `KeyRequest` from the segment metadata, call `require_key`, and decrypt with AES-128 (CTR or CBC per the segment's mode field) using the `aes`+`cbc` crates. Reassemble the inner ELF. **The decryption is standard AES given a supplied key — no key material, no key derivation.**

**Steps:**
- [ ] Write failing tests: (a) a synthetic **plaintext** SELF (single passthrough segment wrapping a minimal ELF) → `decrypt_self(.., &NoKeysProvider)` returns the ELF, `passthrough_segments == 1`. (b) a synthetic SELF with one segment flagged encrypted, AES-CTR-encrypted under a known test key → a stub `KeyProvider` returning that key recovers the original plaintext ELF. (c) the same encrypted SELF with `NoKeysProvider` → `Err(MissingKey { .. })`, no panic.
- [ ] Implement `decrypt_self` (bounds-checked header/entry parse; passthrough vs. decrypt branch; AES-CTR/CBC via `aes`/`cbc`).
- [ ] Run tests; confirm pass. Confirm no panic on a truncated SELF (add a truncated-input test → `Err`, not panic).
- [ ] Commit.

---

### Task 2: `.sprx` module parser (`sprx.rs`)

**Files:**
- Create: `crates/xps5x-firmware/src/sprx.rs`
- Modify: `crates/xps5x-firmware/src/lib.rs` (`pub mod sprx;` + re-exports)
- Test: in-module

**Interfaces:**
- Consumes: `xps5x_loader::elf::parse_elf` / the ELF program headers; the `DecryptedSelf.elf` bytes from Task 1. Recognizes `ET_SCE_DYNAMIC` (0xFE18), `ET_SCE_DYNEXEC` (0xFE10), and program-header types `PT_SCE_DYNLIBDATA` (0x61000000), `PT_SCE_PROCPARAM` (0x61000001), `PT_SCE_MODULE_PARAM` (0x61000002), `PT_SCE_RELRO` (0x61000010) — all already named in `xps5x-loader/src/elf.rs`.
- Produces:
  ```rust
  pub struct SprxModule {
      pub name: String,               // from module param / filename
      pub e_type: u16,
      pub segments: Vec<SprxSegment>, // loadable PT_LOAD segments (vaddr, data, flags)
      pub dynlib_data: Option<Vec<u8>>, // raw PT_SCE_DYNLIBDATA bytes for dynlib::parse
      pub relro: Option<SprxSegment>,
  }
  pub fn parse_sprx(elf: &[u8]) -> Result<SprxModule, FirmwareError>;
  ```

**Steps:**
- [ ] Write failing test: a synthetic ELF with `e_type = ET_SCE_DYNAMIC`, one `PT_LOAD`, and one `PT_SCE_DYNLIBDATA` program header pointing at a small blob → `parse_sprx` returns a `SprxModule` with one segment and `dynlib_data == Some(blob)`.
- [ ] Implement `parse_sprx` (locate program headers, collect `PT_LOAD` segments bounds-checked, extract the `PT_SCE_DYNLIBDATA` slice). Reject a non-SCE `e_type` with `MalformedDynlibData` (out of LM1 scope).
- [ ] Run tests; add a malformed/truncated-phdr test → `Err`, no panic. Commit.

---

### Task 3: `PT_SCE_DYNLIBDATA` decoder (`dynlib/mod.rs`)

**Files:**
- Modify: `crates/xps5x-firmware/src/dynlib/mod.rs` (currently only declares `pub mod nid;`)
- Test: in-module + hand-built dynlibdata fixtures

**Interfaces:**
- Produces:
  ```rust
  pub struct DynlibData {
      pub imports: Vec<SymbolRef>,   // NID + library/module index
      pub exports: Vec<SymbolExport>,// NID + export address (module vaddr)
      pub relocations: Vec<SceRela>, // jmprel + rela merged
      pub needed_modules: Vec<String>,
  }
  pub struct SymbolRef  { pub nid: u64, pub module_index: u16, pub library_index: u16 }
  pub struct SymbolExport { pub nid: u64, pub value: u64 }
  pub struct SceRela    { pub offset: u64, pub info: u64, pub addend: i64 } // r_type = info & 0xffffffff, sym = info >> 32
  pub fn parse_dynlibdata(blob: &[u8], dyn_tags: &[(u64,u64)]) -> Result<DynlibData, FirmwareError>;
  ```

**Approach:** `PT_SCE_DYNLIBDATA` holds a fingerprint symbol table, a string table (encoded NIDs + module/library names), and SCE relocation tables (`DT_SCE_JMPREL`/`DT_SCE_RELA` + their sizes/pltrel), addressed by SCE dynamic tags. Parse the dynamic tags to find the offsets/sizes of the string table, symbol table, and relocation tables **within the blob**, then decode each. NIDs are the custom-base64 encoded names in the string table (use `dynlib::nid::decode_nid`). Every offset/size is bounds-checked against `blob`.

> **RE note (design §7 open item):** exact tag numbering and symbol-record layout are community-RE-derived and may need iteration. Implement the documented common set; where a record field is uncertain, decode what is structurally certain (NID, module/library index, reloc offset/info/addend) and leave a `// TODO(LM1+): verify against real module` with a test asserting the certain fields. Do not guess silently — log unknowns.

**Steps:**
- [ ] Write failing tests over a hand-built dynlibdata blob (string table with two known NID-encoded names, one import record, one export record, one RELA entry) → expected `imports/exports/relocations`.
- [ ] Implement `parse_dynlibdata` (tag walk → table slices → decode, all bounds-checked).
- [ ] Run tests; add a malformed-tag/out-of-bounds-offset test → `Err(MalformedDynlibData)`, no panic. Commit.

---

### Task 4: NID↔name database (`dynlib/nid.rs` extension)

**Files:**
- Modify: `crates/xps5x-firmware/src/dynlib/nid.rs` (has `nid_of`/`encode_nid`/`decode_nid` + salt/alphabet)
- Test: in-module

**Interfaces:**
- Produces:
  ```rust
  /// Maps import NIDs back to the HLE "library::function" names they resolve
  /// to, precomputed from the names XPS5X's HLE implements.
  pub struct NidDatabase { by_nid: HashMap<u64, String> }
  impl NidDatabase {
      pub fn from_hle_names(names: impl IntoIterator<Item = (String, String)>) -> Self; // (library, function)
      pub fn resolve(&self, nid: u64) -> Option<&str>; // -> "library::function"
  }
  ```

**Approach:** For every `(library, function)` the HLE registers, compute `nid_of(function)` and map it to `"library::function"`. This lets an import-by-NID resolve to an HLE function name with no hand-mapping. Seed a small table of well-known names to cover functions the HLE registers by verifying with a public NID vector.

**Steps:**
- [ ] Write failing test: `NidDatabase::from_hle_names([("libkernel","sceKernelAllocDirectMemory")])`; `db.resolve(nid_of("sceKernelAllocDirectMemory"))` → `Some("libkernel::sceKernelAllocDirectMemory")`.
- [ ] **Pin the NID bit-order (design open item):** add a test asserting `encode_nid(nid_of(name))` matches a documented public `name → NID` vector, confirming our NID matches Sony's. If it mismatches, fix `nid.rs` byte order before proceeding.
- [ ] Implement `NidDatabase`. Run tests; commit.

---

### Task 5: `ModuleRegistry` — HLE/LLE dispatch (`registry.rs`)

**Files:**
- Create: `crates/xps5x-firmware/src/registry.rs`
- Modify: `crates/xps5x-firmware/src/lib.rs`
- Test: in-module

**Interfaces:**
- Consumes: `NidDatabase` (Task 4); `xps5x_hle::HleRegistry` (`is_implemented(lib, fn)`); `DynlibData` exports (Task 3).
- Produces:
  ```rust
  pub enum Resolver { Hle { library: String, function: String }, Lle { addr: u64 }, Unresolved }
  pub enum ModulePolicy { PreferHle, PreferLle }
  pub struct ModuleRegistry { /* nid db, hle handle, per-module policy, loaded exports */ }
  impl ModuleRegistry {
      pub fn new(nid_db: NidDatabase) -> Self;
      pub fn set_policy(&mut self, module: &str, policy: ModulePolicy);
      pub fn register_module_exports(&mut self, module: &str, exports: &[SymbolExport]);
      pub fn resolve(&self, hle: &HleRegistry, importing_module: &str, nid: u64) -> Resolver;
  }
  ```

**Approach:** `resolve` consults per-module policy (default `PreferHle`). PreferHle: if the NID maps (via `NidDatabase`) to an HLE-implemented `library::function`, return `Hle`; else try a loaded LLE export; else `Unresolved`. PreferLle: reverse order. Default policy for all modules is `PreferHle` (works today).

**Steps:**
- [ ] Write failing tests: NID of an HLE-implemented function → `Resolver::Hle{..}`; a NID present only as another module's export → `Resolver::Lle{addr}`; an unknown NID → `Resolver::Unresolved`; `set_policy(PreferLle)` flips precedence.
- [ ] Implement `ModuleRegistry`. Run tests; commit.

---

### Task 6: SCE relocation applier / linker (`dynlib/linker.rs`)

**Files:**
- Create: `crates/xps5x-firmware/src/dynlib/linker.rs`
- Modify: `crates/xps5x-firmware/src/dynlib/mod.rs`
- Test: in-module

**Interfaces:**
- Consumes: `SprxModule` segments, `DynlibData` (relocs + imports), `ModuleRegistry`, `HleRegistry`.
- Produces:
  ```rust
  pub struct LinkedModule { pub image: Vec<u8>, pub base: u64, pub unresolved: Vec<u64> /* NIDs */ }
  pub fn link_module(module: &SprxModule, dynlib: &DynlibData,
                     registry: &ModuleRegistry, hle: &HleRegistry, base: u64)
      -> Result<LinkedModule, FirmwareError>;
  ```

**Approach:** Lay the `PT_LOAD` segments into a flat image at `base`. For each relocation, compute the target slot and, for symbol relocations, resolve the symbol's NID through the `ModuleRegistry`. Patch the slot: for an `Hle` resolution, write a synthetic HLE-trampoline address (a deterministic tagged address the runtime will trap and dispatch — LM1 needs a *linked* state, not real execution); for `Lle`, write the export address + addend; for `Unresolved`, write a diagnostic stub address and record the NID. Handle the common SCE relocation types (`R_X86_64_RELATIVE`, `R_X86_64_64`, `R_X86_64_GLOB_DAT`, `R_X86_64_JUMP_SLOT`); an unhandled type → `FirmwareError::UnsupportedRelocation(r_type)`.

**Steps:**
- [ ] Write failing tests: a module with one RELATIVE reloc → slot = base + addend; one JUMP_SLOT import whose NID is HLE-resolved → slot = the deterministic HLE-trampoline address; one import with an unknown NID → slot = stub addr and NID recorded in `unresolved`; an unsupported reloc type → `Err(UnsupportedRelocation)`.
- [ ] Implement `link_module`. Run tests; commit.

---

### Task 7: End-to-end pipeline test + `Firmware`/CLI wiring (LM1 acceptance)

**Files:**
- Create: `crates/xps5x-firmware/tests/homebrew_pipeline.rs`
- Modify: `crates/xps5x-firmware/src/lib.rs` (a `load_module(elf_or_self, provider, registry, hle)` convenience that chains Tasks 1→6); optionally `crates/xps5x-gui/src/main.rs` (a `--load-sprx <path>` diagnostic mirroring `--firmware-info`).
- Test: integration test

**Interfaces:**
- Produces:
  ```rust
  pub fn load_module(bytes: &[u8], provider: &dyn KeyProvider,
                     registry: &ModuleRegistry, hle: &HleRegistry, base: u64)
      -> Result<LinkedModule, FirmwareError>; // SELF? decrypt/passthrough → sprx → dynlib → link
  ```

**Steps:**
- [ ] Build a **synthetic homebrew `.sprx`** in the test: a plaintext SELF wrapping an `ET_SCE_DYNAMIC` ELF with one `PT_LOAD`, a `PT_SCE_DYNLIBDATA` declaring one import (NID of an HLE-registered function) and one JUMP_SLOT reloc against it.
- [ ] Write the failing acceptance test: `load_module(sprx, &NoKeysProvider, &registry, &hle, base)` returns a `LinkedModule` whose import slot holds the HLE-trampoline address and whose `unresolved` is empty. A second variant with an unknown-NID import → the NID appears in `unresolved` and the call does **not** fail (logged, non-fatal).
- [ ] Implement `load_module`; wire the optional `--load-sprx` CLI diagnostic (prints module name, import/export counts, resolved/unresolved tallies; never decrypts without a provider).
- [ ] Run `cargo test --workspace`; confirm the pipeline test passes. Commit.

---

## LM1 acceptance (design §6)

- A homebrew/decrypted `.sprx` loads, links against HLE stubs, and reaches a defined linked state; unresolved imports are logged, not fatal. Verified by `tests/homebrew_pipeline.rs` with entirely synthetic inputs.
- `NoKeysProvider` path is clean throughout; a stub provider round-trips a synthetic encrypted segment.
- No key material or firmware blobs anywhere in the crate or tests; clippy + rustfmt clean workspace-wide.
