# LLE Firmware Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `raeen-firmware` crate that ingests PS5 firmware, decrypts SELF through a user-supplied key boundary, and loads + NID-links Sony's real userland modules against either HLE reimplementations or other real modules (LM0 → LM1).

**Architecture:** A new `raeen-firmware` crate sits on top of `raeen-loader` (ELF/SELF primitives) and `raeen-core` (errors, types, the `KeyProvider` trait). It adds a PUP container parser, a `.sprx`/SCE-module parser, a `PT_SCE_DYNLIBDATA` decoder, a NID-based dynamic linker, and a `ModuleRegistry` that dispatches each import to an HLE function (from `raeen-hle`) or an exported symbol of another loaded module. Decryption machinery is added to `raeen-loader::self_format`, driven entirely by a `KeyProvider`; the default provider supplies no keys.

**Tech Stack:** Rust (edition 2024, rustc 1.85), `goblin` (ELF), `aes` + `cbc` (SELF segment decryption), `sha1` + `base64` (NID hashing), `tracing`, `thiserror`.

## Global Constraints

- **Rust edition `2024`, `rust-version = "1.85"`** — copied from `[workspace.package]`. New crate inherits via `.workspace = true`.
- **License `GPL-2.0-only`** on every new crate (`license.workspace = true`).
- **Clean-room boundary (from spec §2):** Raeen ships **no keys, no firmware, no key-extraction/circumvention code.** The only decryption path consumes keys from a user-supplied `KeyProvider`. Never commit key material or firmware bytes. Never add code that derives, brute-forces, glitches, or extracts keys. All crypto tests use synthetic keys/data the test itself generates.
- **All dependencies via `{ workspace = true }`** — add new external crates to root `[workspace.dependencies]` first, then reference them.
- **Commit convention:** Conventional Commits (`feat:`, `test:`, `refactor:`, `chore:`). Every commit message ends with the trailer:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```
  (Shown in Task 1's commit; apply to every commit in this plan.)
- **Do not commit the firmware blob.** `PS5 Firmware/` is already `.gitignore`d; keep it that way.
- **Test command baseline:** `cargo test -p <crate>` for a single crate; `cargo test --workspace` before the final commit of each task that touches multiple crates.

---

## File Structure

**New crate `crates/raeen-firmware/`:**
- `Cargo.toml` — crate manifest + `firmware-info` bin target.
- `src/lib.rs` — public API surface; module declarations (grown task-by-task).
- `src/pup.rs` — PUP container parser (`Firmware`, `PupEntry`, `PupHeader`).
- `src/sprx.rs` — SCE module parser (`Module`, extracts `PT_SCE_DYNLIBDATA`).
- `src/dynlib/mod.rs` — `PT_SCE_DYNLIBDATA` decoder (imports/exports/relocs).
- `src/dynlib/nid.rs` — NID hashing + `NidDatabase`.
- `src/dynlib/linker.rs` — NID linker / relocation applier (`link_module`).
- `src/registry.rs` — `ModuleRegistry`, per-module HLE/LLE policy.
- `src/loader.rs` — top-level `ModuleLoader::load_module` orchestration.
- `src/bin/firmware-info.rs` — `--firmware-info` diagnostic binary.

**Modified existing files:**
- `Cargo.toml` (root) — add crate to members + `[workspace.dependencies]`; add `aes`, `cbc`, `sha1`, `base64`.
- `crates/raeen-core/src/error.rs` — add `FirmwareError`, wire into `RaeenError`.
- `crates/raeen-core/src/lib.rs` — add `pub mod crypto;`.
- `crates/raeen-core/src/crypto.rs` — **new**: `KeyProvider`, `KeyRequest`, `SegmentKey`, `NoKeysProvider`.
- `crates/raeen-loader/Cargo.toml` — add `aes`, `cbc`.
- `crates/raeen-loader/src/self_format.rs` — add `parse_self_with_provider`, AES-CBC decrypt; route `parse_self` through it.
- `crates/raeen-hle/src/lib.rs` — add `HleRegistry::entries()` enumerator.
- `crates/raeen-hle/Cargo.toml` — (unchanged; firmware depends on hle, not vice-versa).

---

## Task 1: Scaffold `raeen-firmware` crate + `FirmwareError`

**Files:**
- Modify: `Cargo.toml` (root) — members + workspace deps
- Modify: `crates/raeen-core/src/error.rs`
- Create: `crates/raeen-firmware/Cargo.toml`
- Create: `crates/raeen-firmware/src/lib.rs`

**Interfaces:**
- Produces: crate `raeen-firmware` (buildable); `raeen_core::error::FirmwareError` enum with variants `InvalidPupMagic(u32)`, `PupEntryOutOfBounds { index: usize }`, `MissingKey { key_id: u64 }`, `UnsupportedRelocation(u32)`, `MalformedDynlibData(String)`, `Loader(#[from] LoaderError)`; `RaeenError::Firmware(#[from] FirmwareError)`.

- [ ] **Step 1: Add external deps + new crate to root `Cargo.toml`**

In `[workspace.dependencies]`, after the `zstd = "0.13"` line, add:
```toml
# Firmware crypto + NID hashing
aes = "0.8"
cbc = "0.10"
sha1 = "0.10"
base64 = "0.22"
```
In the same section, after `raeen-gui = { path = "crates/raeen-gui" }`, add:
```toml
raeen-firmware = { path = "crates/raeen-firmware" }
```
In `[workspace] members`, after `"crates/raeen-gui",` add:
```toml
    "crates/raeen-firmware",
```

- [ ] **Step 2: Add `FirmwareError` to `crates/raeen-core/src/error.rs`**

Add a new variant to `RaeenError` (after the `Io` variant, before `Config`):
```rust
    /// Firmware ingestion errors (PUP, module loading, dynamic linking).
    #[error("Firmware error: {0}")]
    Firmware(#[from] FirmwareError),
```
Add this enum at the end of the file:
```rust
/// Errors from the firmware ingestion subsystem (PUP, SELF decryption seam,
/// SCE module loading, NID linking).
#[derive(Debug, Error)]
pub enum FirmwareError {
    #[error("Invalid PUP magic: got {0:#010x}")]
    InvalidPupMagic(u32),

    #[error("PUP entry {index} extends beyond file bounds")]
    PupEntryOutOfBounds { index: usize },

    #[error("No key available for key_id {key_id:#x} (user-supplied KeyProvider returned none)")]
    MissingKey { key_id: u64 },

    #[error("Unsupported SCE relocation type: {0:#x}")]
    UnsupportedRelocation(u32),

    #[error("Malformed PT_SCE_DYNLIBDATA: {0}")]
    MalformedDynlibData(String),

    #[error("Loader error: {0}")]
    Loader(#[from] LoaderError),
}
```

- [ ] **Step 3: Create `crates/raeen-firmware/Cargo.toml`**

```toml
[package]
name = "raeen-firmware"
description = "PS5 firmware ingestion — PUP parsing, SELF decryption seam, SCE module loading, NID linking"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
raeen-core = { workspace = true }
raeen-loader = { workspace = true }
raeen-hle = { workspace = true }
goblin = { workspace = true }
sha1 = { workspace = true }
base64 = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
anyhow = { workspace = true }

[[bin]]
name = "firmware-info"
path = "src/bin/firmware-info.rs"
```

- [ ] **Step 4: Create `crates/raeen-firmware/src/lib.rs`**

```rust
//! # Raeen Firmware
//!
//! The "firmware spine": ingests PS5 firmware packages (PUP), decrypts SELF
//! modules through a user-supplied [`raeen_core::crypto::KeyProvider`], parses
//! Sony's SCE dynamic modules, and links their imports by NID against either
//! HLE reimplementations (`raeen-hle`) or exports of other loaded modules.
//!
//! ## Clean-room boundary
//!
//! This crate contains **no keys, no firmware, and no key-extraction code.**
//! Decryption consumes keys from a `KeyProvider` the user supplies from
//! hardware they own. The default provider supplies nothing and decryption
//! fails cleanly.

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        assert!(!super::VERSION.is_empty());
    }
}
```

- [ ] **Step 5: Verify the workspace builds and the crate test passes**

Run: `cargo test -p raeen-firmware -p raeen-core`
Expected: compiles; `crate_builds` passes; `raeen-core` still passes.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/raeen-core/src/error.rs crates/raeen-firmware/
git commit -m "$(cat <<'EOF'
feat(firmware): scaffold raeen-firmware crate and FirmwareError

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: PUP container parser

**Files:**
- Create: `crates/raeen-firmware/src/pup.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod pup;`)

**Interfaces:**
- Consumes: `raeen_core::error::FirmwareError`.
- Produces:
  - `pub struct PupHeader { pub magic: u32, pub version: u16, pub entry_count: u16 }`
  - `pub struct PupEntry { pub id: u32, pub offset: u64, pub size: u64, pub flags: u32 }`
  - `pub struct Firmware { pub header: PupHeader, pub entries: Vec<PupEntry>, data: Vec<u8> }`
  - `Firmware::from_bytes(data: Vec<u8>) -> Result<Firmware, FirmwareError>`
  - `Firmware::open(path: impl AsRef<Path>) -> Result<Firmware, FirmwareError>`
  - `Firmware::entries(&self) -> &[PupEntry]`
  - `Firmware::read_entry(&self, entry: &PupEntry) -> Result<&[u8], FirmwareError>` (returns raw, still-encrypted bytes)

> **Note on real-format confirmation:** The byte-exact PS5 PUP layout is confirmed empirically by running the Task 3 diagnostic against the real `PS5UPDATE.PUP`. The parser below reads a documented-shape container header and entry table; `PUP_MAGIC` and field offsets are **updated after Task 3** if the real file differs. Unit tests here use synthetic fixtures the test constructs, so they remain correct regardless of the real magic.

- [ ] **Step 1: Write the failing tests**

Create `crates/raeen-firmware/src/pup.rs`:
```rust
//! PUP (PlayStation Update Package) container parser.
//!
//! Parses the outer container structure and entry table to the extent it is
//! unencrypted. Encrypted payloads are returned as raw bytes; decryption is a
//! separate, explicit, key-gated step (see `raeen_loader::self_format`).

use std::path::Path;
use tracing::{debug, info};
use raeen_core::error::FirmwareError;

/// Outer PUP container magic. Confirmed empirically against the real firmware
/// via the `firmware-info` diagnostic; update if the real file differs.
pub const PUP_MAGIC: u32 = 0x1D3D154F;

/// Size of the fixed PUP header we parse (magic + version + entry_count + pad).
const PUP_HEADER_SIZE: usize = 16;
/// Size of one entry-table record.
const PUP_ENTRY_SIZE: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PupHeader {
    pub magic: u32,
    pub version: u16,
    pub entry_count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PupEntry {
    pub id: u32,
    pub offset: u64,
    pub size: u64,
    pub flags: u32,
}

#[derive(Debug)]
pub struct Firmware {
    pub header: PupHeader,
    pub entries: Vec<PupEntry>,
    data: Vec<u8>,
}

impl Firmware {
    pub fn from_bytes(data: Vec<u8>) -> Result<Firmware, FirmwareError> {
        if data.len() < PUP_HEADER_SIZE {
            return Err(FirmwareError::InvalidPupMagic(0));
        }
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != PUP_MAGIC {
            return Err(FirmwareError::InvalidPupMagic(magic));
        }
        let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
        let entry_count = u16::from_le_bytes(data[6..8].try_into().unwrap());
        let header = PupHeader { magic, version, entry_count };
        info!("PUP: version={version}, entries={entry_count}");

        let mut entries = Vec::with_capacity(entry_count as usize);
        for i in 0..entry_count as usize {
            let base = PUP_HEADER_SIZE + i * PUP_ENTRY_SIZE;
            if base + PUP_ENTRY_SIZE > data.len() {
                return Err(FirmwareError::PupEntryOutOfBounds { index: i });
            }
            let id = u32::from_le_bytes(data[base..base + 4].try_into().unwrap());
            let offset = u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
            let size = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
            let flags = u32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap());
            let entry = PupEntry { id, offset, size, flags };
            debug!("  entry {i}: id={id:#x} offset={offset:#x} size={size:#x}");
            entries.push(entry);
        }
        Ok(Firmware { header, entries, data })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Firmware, FirmwareError> {
        let bytes = std::fs::read(path).map_err(|e| {
            FirmwareError::Loader(raeen_core::error::LoaderError::Io(e))
        })?;
        Firmware::from_bytes(bytes)
    }

    pub fn entries(&self) -> &[PupEntry] {
        &self.entries
    }

    pub fn read_entry(&self, entry: &PupEntry) -> Result<&[u8], FirmwareError> {
        let start = entry.offset as usize;
        let end = start
            .checked_add(entry.size as usize)
            .ok_or(FirmwareError::PupEntryOutOfBounds { index: 0 })?;
        if end > self.data.len() {
            return Err(FirmwareError::PupEntryOutOfBounds { index: 0 });
        }
        Ok(&self.data[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic PUP: header + `entries.len()` records + payload region.
    fn synthetic_pup(entries: &[PupEntry]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&PUP_MAGIC.to_le_bytes()); // magic
        buf.extend_from_slice(&1u16.to_le_bytes()); // version
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes()); // entry_count
        buf.extend_from_slice(&[0u8; 6]); // pad to 16
        for e in entries {
            buf.extend_from_slice(&e.id.to_le_bytes());
            buf.extend_from_slice(&e.flags.to_le_bytes());
            buf.extend_from_slice(&e.offset.to_le_bytes());
            buf.extend_from_slice(&e.size.to_le_bytes());
        }
        // Extend to cover the highest referenced payload region.
        let max_end = entries
            .iter()
            .map(|e| e.offset as usize + e.size as usize)
            .max()
            .unwrap_or(0);
        if buf.len() < max_end {
            buf.resize(max_end, 0xAB);
        }
        buf
    }

    #[test]
    fn parses_header_and_entries() {
        let entries = [
            PupEntry { id: 0x100, offset: 64, size: 4, flags: 0 },
            PupEntry { id: 0x200, offset: 68, size: 8, flags: 1 },
        ];
        let bytes = synthetic_pup(&entries);
        let fw = Firmware::from_bytes(bytes).unwrap();
        assert_eq!(fw.header.entry_count, 2);
        assert_eq!(fw.entries().len(), 2);
        assert_eq!(fw.entries()[0].id, 0x100);
        assert_eq!(fw.entries()[1].size, 8);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0u8; 32];
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let err = Firmware::from_bytes(bytes).unwrap_err();
        assert!(matches!(err, FirmwareError::InvalidPupMagic(0xDEADBEEF)));
    }

    #[test]
    fn rejects_truncated_entry_table() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PUP_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&5u16.to_le_bytes()); // claims 5 entries
        bytes.extend_from_slice(&[0u8; 6]);
        // ...but provides no entry records.
        let err = Firmware::from_bytes(bytes).unwrap_err();
        assert!(matches!(err, FirmwareError::PupEntryOutOfBounds { index: 0 }));
    }

    #[test]
    fn read_entry_returns_payload_slice() {
        let entries = [PupEntry { id: 1, offset: 40, size: 4, flags: 0 }];
        let mut bytes = synthetic_pup(&entries);
        bytes[40..44].copy_from_slice(&[1, 2, 3, 4]);
        let fw = Firmware::from_bytes(bytes).unwrap();
        assert_eq!(fw.read_entry(&fw.entries()[0]).unwrap(), &[1, 2, 3, 4]);
    }
}
```

- [ ] **Step 2: Declare the module**

In `crates/raeen-firmware/src/lib.rs`, add after the `VERSION` const:
```rust
pub mod pup;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p raeen-firmware pup`
Expected: 4 `pup::tests::*` pass.

- [ ] **Step 4: Commit**

```bash
git add crates/raeen-firmware/src/pup.rs crates/raeen-firmware/src/lib.rs
git commit -m "$(printf 'feat(firmware): add PUP container parser\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 3: `firmware-info` diagnostic binary (LM0 acceptance)

**Files:**
- Create: `crates/raeen-firmware/src/bin/firmware-info.rs`

**Interfaces:**
- Consumes: `raeen_firmware::pup::Firmware`.
- Produces: binary `firmware-info` runnable via `cargo run -p raeen-firmware --bin firmware-info -- <path>`.

- [ ] **Step 1: Write an integration test for the library path the binary uses**

Create `crates/raeen-firmware/tests/firmware_info.rs`:
```rust
//! Exercises the same `Firmware::open` path the `firmware-info` binary uses.

use std::io::Write;
use raeen_firmware::pup::{Firmware, PUP_MAGIC};

#[test]
fn opens_a_synthetic_pup_file_from_disk() {
    let dir = std::env::temp_dir();
    let path = dir.join("raeen_test_synthetic.pup");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&PUP_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // version
    bytes.extend_from_slice(&1u16.to_le_bytes()); // entry_count
    bytes.extend_from_slice(&[0u8; 6]); // pad
    bytes.extend_from_slice(&0x100u32.to_le_bytes()); // id
    bytes.extend_from_slice(&0u32.to_le_bytes()); // flags
    bytes.extend_from_slice(&64u64.to_le_bytes()); // offset
    bytes.extend_from_slice(&4u64.to_le_bytes()); // size
    bytes.resize(68, 0);

    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&bytes).unwrap();
    drop(f);

    let fw = Firmware::open(&path).unwrap();
    assert_eq!(fw.entries().len(), 1);
    assert_eq!(fw.entries()[0].id, 0x100);

    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p raeen-firmware --test firmware_info`
Expected: FAIL to compile (`can't find crate` is not it — the lib exists; it should actually PASS if `pub mod pup` is exported). If it passes, that is fine — proceed; the binary is still needed for LM0. If it fails because `Firmware::open` is private, make the method `pub` (it is already `pub` per Task 2).

- [ ] **Step 3: Create the binary**

Create `crates/raeen-firmware/src/bin/firmware-info.rs`:
```rust
//! `firmware-info` — enumerate the structure of a PS5 firmware package.
//!
//! Usage: firmware-info <path-to-PUP>
//!
//! Reports the container header and entry table. Does NOT decrypt anything;
//! encrypted payloads are reported as-is.

use std::process::ExitCode;
use raeen_firmware::pup::Firmware;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: firmware-info <path-to-PUP>");
            return ExitCode::from(2);
        }
    };

    match Firmware::open(&path) {
        Ok(fw) => {
            println!("PUP: {path}");
            println!("  magic       : {:#010x}", fw.header.magic);
            println!("  version     : {}", fw.header.version);
            println!("  entry_count : {}", fw.header.entry_count);
            println!("  entries:");
            for (i, e) in fw.entries().iter().enumerate() {
                println!(
                    "    [{i:>3}] id={:#010x} offset={:#012x} size={:#012x} flags={:#x}",
                    e.id, e.offset, e.size, e.flags
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
```

- [ ] **Step 4: Add the binary's own deps**

The binary uses `tracing_subscriber`. Add it to `crates/raeen-firmware/Cargo.toml` under `[dependencies]`:
```toml
tracing-subscriber = { workspace = true }
```

- [ ] **Step 5: Verify the binary builds and the test passes**

Run: `cargo test -p raeen-firmware --test firmware_info`
Then: `cargo build -p raeen-firmware --bin firmware-info`
Expected: both succeed.

- [ ] **Step 6: Manual LM0 acceptance against the real firmware**

Run: `cargo run -p raeen-firmware --bin firmware-info -- "PS5 Firmware/PS5UPDATE.PUP"`
Expected: prints a magic value and (if the real layout matches) an entry table. **If the magic differs from `PUP_MAGIC`,** note the printed value, update `PUP_MAGIC` and, if needed, the header/entry offsets in `pup.rs`, and re-run Task 2's tests (they use synthetic fixtures, so they still pass). It is acceptable for LM0 if the tool cleanly reports the real magic and does not crash — full real-layout decoding is iterated as the format is confirmed.

- [ ] **Step 7: Commit**

```bash
git add crates/raeen-firmware/
git commit -m "$(printf 'feat(firmware): add firmware-info diagnostic binary (LM0)\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 4: `KeyProvider` trait + `NoKeysProvider` in `raeen-core`

**Files:**
- Create: `crates/raeen-core/src/crypto.rs`
- Modify: `crates/raeen-core/src/lib.rs` (add `pub mod crypto;`)

**Interfaces:**
- Produces:
  - `pub struct KeyRequest { pub key_type: u32, pub key_id: u64, pub segment_index: u64 }`
  - `pub struct SegmentKey { pub key: [u8; 16], pub iv: [u8; 16] }`
  - `pub trait KeyProvider: Send + Sync { fn segment_key(&self, req: &KeyRequest) -> Option<SegmentKey>; }`
  - `pub struct NoKeysProvider;` implementing `KeyProvider` (always returns `None`).

- [ ] **Step 1: Write the failing test**

Create `crates/raeen-core/src/crypto.rs`:
```rust
//! The decryption key boundary.
//!
//! Raeen never contains keys. A [`KeyProvider`] is supplied by the user from
//! keys they obtained from hardware they own. The default [`NoKeysProvider`]
//! supplies nothing, so encrypted content fails to decrypt cleanly.

/// Identifies which key is needed to decrypt a SELF/module segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRequest {
    /// SELF `key_type` field.
    pub key_type: u32,
    /// Key identifier / seed from the SELF metadata.
    pub key_id: u64,
    /// Segment index within the SELF.
    pub segment_index: u64,
}

/// A content key + IV for one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentKey {
    pub key: [u8; 16],
    pub iv: [u8; 16],
}

/// Supplies content keys for encrypted SELF/module segments.
///
/// Implementors are user-provided. Raeen ships only [`NoKeysProvider`].
pub trait KeyProvider: Send + Sync {
    fn segment_key(&self, req: &KeyRequest) -> Option<SegmentKey>;
}

/// The default provider: no keys, ever. Encrypted content cannot be decrypted.
pub struct NoKeysProvider;

impl KeyProvider for NoKeysProvider {
    fn segment_key(&self, _req: &KeyRequest) -> Option<SegmentKey> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_keys_provider_returns_none() {
        let p = NoKeysProvider;
        let req = KeyRequest { key_type: 0, key_id: 0, segment_index: 0 };
        assert!(p.segment_key(&req).is_none());
    }

    #[test]
    fn key_provider_is_object_safe() {
        // Compiles only if the trait is object-safe (needed for &dyn KeyProvider).
        let p: &dyn KeyProvider = &NoKeysProvider;
        let req = KeyRequest { key_type: 1, key_id: 2, segment_index: 3 };
        assert!(p.segment_key(&req).is_none());
    }
}
```

- [ ] **Step 2: Declare the module**

In `crates/raeen-core/src/lib.rs`, add to the module list (after `pub mod config;`):
```rust
pub mod crypto;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p raeen-core crypto`
Expected: `no_keys_provider_returns_none` and `key_provider_is_object_safe` pass.

- [ ] **Step 4: Commit**

```bash
git add crates/raeen-core/src/crypto.rs crates/raeen-core/src/lib.rs
git commit -m "$(printf 'feat(core): add KeyProvider decryption boundary\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 5: SELF decryption seam in `raeen-loader`

**Files:**
- Modify: `crates/raeen-loader/Cargo.toml`
- Modify: `crates/raeen-loader/src/self_format.rs`

**Interfaces:**
- Consumes: `raeen_core::crypto::{KeyProvider, KeyRequest, SegmentKey, NoKeysProvider}`.
- Produces:
  - `pub fn parse_self_with_provider(data: &[u8], provider: &dyn KeyProvider) -> Result<LoadedBinary, LoaderError>`
  - `parse_self(data: &[u8])` becomes a thin wrapper calling `parse_self_with_provider(data, &NoKeysProvider)`.
  - `fn decrypt_aes128_cbc(data: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Result<Vec<u8>, LoaderError>` (module-private).

- [ ] **Step 1: Add crypto deps to the loader**

In `crates/raeen-loader/Cargo.toml` `[dependencies]`, add:
```toml
aes = { workspace = true }
cbc = { workspace = true }
```

- [ ] **Step 2: Write the failing tests**

In `crates/raeen-loader/src/self_format.rs`, replace the existing `#[cfg(test)] mod tests { ... }` block with this expanded version:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use aes::cipher::{BlockEncryptMut, KeyIvInit};
    use raeen_core::crypto::{KeyProvider, KeyRequest, SegmentKey};

    type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;

    /// A minimal, goblin-parseable 64-byte ELF64 x86-64 header (no program headers).
    fn minimal_elf(entry: u64) -> Vec<u8> {
        let mut e = vec![0u8; 64];
        e[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        e[4] = 2; // ELFCLASS64
        e[5] = 1; // little-endian
        e[6] = 1; // EV_CURRENT
        e[16..18].copy_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
        e[18..20].copy_from_slice(&0x3Eu16.to_le_bytes()); // e_machine = x86-64
        e[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
        e[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry
        e[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
        e
    }

    /// Build a SELF whose inner region (at header_size) is `inner`, with one
    /// entry marked encrypted/has-data iff `encrypted`.
    fn build_self(inner: &[u8], encrypted: bool) -> Vec<u8> {
        let header_size: u16 = 64;
        let mut buf = vec![0u8; header_size as usize];
        buf[0..4].copy_from_slice(&SELF_MAGIC.to_le_bytes());
        buf[4] = 1; // version
        buf[5] = 0; // mode = PS5
        buf[6] = 1; // little-endian
        buf[12..14].copy_from_slice(&header_size.to_le_bytes());
        buf[24..26].copy_from_slice(&1u16.to_le_bytes()); // num_entries = 1
        // One entry at offset 32. properties: bit0 unused; encryption in bits1-3;
        // set uncompressed_size so has_data() is true.
        let properties: u64 = if encrypted { 0b010 } else { 0 };
        buf[32..40].copy_from_slice(&properties.to_le_bytes());
        buf[40..48].copy_from_slice(&(header_size as u64).to_le_bytes()); // offset
        buf[48..56].copy_from_slice(&(inner.len() as u64).to_le_bytes()); // compressed_size
        buf[56..64].copy_from_slice(&(inner.len() as u64).to_le_bytes()); // uncompressed_size
        buf.extend_from_slice(inner);
        buf
    }

    struct StubProvider {
        key: SegmentKey,
    }
    impl KeyProvider for StubProvider {
        fn segment_key(&self, _req: &KeyRequest) -> Option<SegmentKey> {
            Some(self.key)
        }
    }

    #[test]
    fn decrypt_aes128_cbc_roundtrips() {
        let key = [0x11u8; 16];
        let iv = [0x22u8; 16];
        let plaintext = minimal_elf(0xCAFE); // 64 bytes, multiple of block size
        let mut ct = plaintext.clone();
        // Encrypt in place, block by block (no padding; length is a multiple of 16).
        let mut enc = Aes128CbcEnc::new(&key.into(), &iv.into());
        for chunk in ct.chunks_exact_mut(16) {
            let block = aes::cipher::generic_array::GenericArray::from_mut_slice(chunk);
            enc.encrypt_block_mut(block);
        }
        let decrypted = decrypt_aes128_cbc(&ct, &key, &iv).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypted_self_without_keys_errors() {
        let inner = minimal_elf(0x1000);
        let self_bytes = build_self(&inner, true);
        let err = parse_self(&self_bytes).unwrap_err();
        assert!(matches!(err, LoaderError::EncryptedSelf));
    }

    #[test]
    fn encrypted_self_with_key_decrypts_inner() {
        let key = [0x33u8; 16];
        let iv = [0x44u8; 16];
        let plaintext_inner = minimal_elf(0xBEEF);
        let mut ct = plaintext_inner.clone();
        let mut enc = Aes128CbcEnc::new(&key.into(), &iv.into());
        for chunk in ct.chunks_exact_mut(16) {
            let block = aes::cipher::generic_array::GenericArray::from_mut_slice(chunk);
            enc.encrypt_block_mut(block);
        }
        let self_bytes = build_self(&ct, true);
        let provider = StubProvider { key: SegmentKey { key, iv } };
        let bin = parse_self_with_provider(&self_bytes, &provider).unwrap();
        assert_eq!(bin.entry_point, 0xBEEF);
    }

    #[test]
    fn decrypted_self_passthrough_still_works() {
        let inner = minimal_elf(0x2000);
        let self_bytes = build_self(&inner, false);
        let bin = parse_self(&self_bytes).unwrap();
        assert_eq!(bin.entry_point, 0x2000);
    }

    #[test]
    fn test_invalid_self_magic() {
        let data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let result = parse_self(&data);
        assert!(matches!(result, Err(LoaderError::InvalidSelfMagic(_))));
    }

    #[test]
    fn test_auto_detect_elf() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        data[4] = 2;
        let result = load_binary(&data);
        assert!(!matches!(result, Err(LoaderError::InvalidSelfMagic(_))));
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p raeen-loader self_format`
Expected: FAIL — `parse_self_with_provider` and `decrypt_aes128_cbc` don't exist yet.

- [ ] **Step 4: Implement the decryption seam**

In `crates/raeen-loader/src/self_format.rs`:

(a) Update the imports at the top of the file:
```rust
use crate::LoadedBinary;
use aes::cipher::{BlockDecryptMut, KeyIvInit};
use tracing::{debug, info, warn};
use raeen_core::crypto::{KeyProvider, KeyRequest, NoKeysProvider};
use raeen_core::error::LoaderError;

type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;
```

(b) Add the private decrypt helper (place it above `parse_self`):
```rust
/// Decrypt an AES-128-CBC region. `data.len()` must be a multiple of 16.
fn decrypt_aes128_cbc(
    data: &[u8],
    key: &[u8; 16],
    iv: &[u8; 16],
) -> Result<Vec<u8>, LoaderError> {
    if data.len() % 16 != 0 {
        return Err(LoaderError::SegmentLoadFailed {
            address: 0,
            size: data.len() as u64,
            reason: "encrypted region is not a multiple of the AES block size".into(),
        });
    }
    let mut out = data.to_vec();
    let mut dec = Aes128CbcDec::new(key.into(), iv.into());
    for chunk in out.chunks_exact_mut(16) {
        let block = aes::cipher::generic_array::GenericArray::from_mut_slice(chunk);
        dec.decrypt_block_mut(block);
    }
    Ok(out)
}
```

(c) Replace the body of `parse_self` so it delegates, and add the provider-aware function. Change the existing `pub fn parse_self(data: &[u8]) -> Result<LoadedBinary, LoaderError> {` implementation to:
```rust
/// Parse a SELF using the default (no-keys) provider. Encrypted SELFs error.
pub fn parse_self(data: &[u8]) -> Result<LoadedBinary, LoaderError> {
    parse_self_with_provider(data, &NoKeysProvider)
}

/// Parse a SELF, decrypting encrypted segments via the supplied `provider`.
///
/// If the SELF has encrypted segments and the provider supplies a key, the
/// inner ELF region is decrypted and parsed. If no key is available, returns
/// `LoaderError::EncryptedSelf` (the normal default path).
pub fn parse_self_with_provider(
    data: &[u8],
    provider: &dyn KeyProvider,
) -> Result<LoadedBinary, LoaderError> {
    if data.len() < std::mem::size_of::<SelfHeader>() {
        return Err(LoaderError::InvalidSelfMagic(0));
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != SELF_MAGIC {
        return Err(LoaderError::InvalidSelfMagic(magic));
    }
    info!("Parsing SELF file ({} bytes)", data.len());

    let version = data[4];
    let mode = data[5];
    let key_type = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let header_size = u16::from_le_bytes([data[12], data[13]]);
    let file_size = u64::from_le_bytes(data[16..24].try_into().unwrap());
    let num_entries = u16::from_le_bytes([data[24], data[25]]);

    debug!(
        "SELF header: version={version}, mode={mode}, entries={num_entries}, \
         header_size={header_size:#x}, file_size={file_size:#x}"
    );

    let entry_offset = 32usize;
    let entry_size = 32usize;
    let mut has_encrypted_segments = false;

    for i in 0..num_entries as usize {
        let base = entry_offset + i * entry_size;
        if base + entry_size > data.len() {
            warn!("SELF entry {i} extends beyond file bounds");
            break;
        }
        let properties = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let compressed_size = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
        let uncompressed_size = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());
        let entry = SelfEntry {
            properties,
            offset: u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap()),
            compressed_size,
            uncompressed_size,
        };
        if entry.is_encrypted() && entry.has_data() {
            has_encrypted_segments = true;
        }
    }

    let elf_offset = header_size as usize;
    if elf_offset >= data.len() {
        return Err(LoaderError::SegmentLoadFailed {
            address: 0,
            size: 0,
            reason: format!(
                "SELF header_size ({elf_offset:#x}) exceeds file size ({:#x})",
                data.len()
            ),
        });
    }

    if has_encrypted_segments {
        let req = KeyRequest { key_type, key_id: 0, segment_index: 0 };
        match provider.segment_key(&req) {
            Some(k) => {
                info!("Decrypting inner ELF region via user-supplied KeyProvider");
                let decrypted = decrypt_aes128_cbc(&data[elf_offset..], &k.key, &k.iv)?;
                return crate::elf::parse_elf(&decrypted);
            }
            None => {
                warn!("SELF has encrypted segments and no key is available");
                return Err(LoaderError::EncryptedSelf);
            }
        }
    }

    info!("Extracting inner ELF from decrypted SELF at offset {elf_offset:#x}");
    crate::elf::parse_elf(&data[elf_offset..])
}
```

(d) Add the loader's dep on `raeen-core::crypto` — no manifest change needed (`raeen-core` is already a dependency); the `use` in (a) is sufficient.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p raeen-loader self_format`
Expected: all six tests pass, including `encrypted_self_with_key_decrypts_inner` and `encrypted_self_without_keys_errors`.

- [ ] **Step 6: Confirm the whole workspace still builds**

Run: `cargo test --workspace`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add crates/raeen-loader/Cargo.toml crates/raeen-loader/src/self_format.rs Cargo.lock
git commit -m "$(printf 'feat(loader): route SELF decryption through KeyProvider seam\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 6: SCE module parser (`sprx.rs`)

**Files:**
- Create: `crates/raeen-firmware/src/sprx.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod sprx;`)

**Interfaces:**
- Consumes: `goblin::elf::Elf`, `FirmwareError`.
- Produces:
  - `pub struct ModuleSegment { pub vaddr: u64, pub mem_size: u64, pub data: Vec<u8>, pub executable: bool }`
  - `pub struct Module { pub name: String, pub entry_point: u64, pub segments: Vec<ModuleSegment>, pub dynlib_data: Vec<u8> }`
  - `pub fn parse_module(data: &[u8]) -> Result<Module, FirmwareError>`

- [ ] **Step 1: Write the failing tests**

Create `crates/raeen-firmware/src/sprx.rs`:
```rust
//! SCE dynamic module (`.sprx` / `ET_SCE_DYNLIB`) parser.
//!
//! Parses the ELF layer via `goblin`, extracts loadable segments, and pulls
//! out the `PT_SCE_DYNLIBDATA` blob for the NID linker.

use tracing::{debug, info};
use raeen_core::error::FirmwareError;

/// PS5-specific program header type carrying the dynamic-linking tables.
const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;

#[derive(Debug, Clone)]
pub struct ModuleSegment {
    pub vaddr: u64,
    pub mem_size: u64,
    pub data: Vec<u8>,
    pub executable: bool,
}

#[derive(Debug)]
pub struct Module {
    pub name: String,
    pub entry_point: u64,
    pub segments: Vec<ModuleSegment>,
    pub dynlib_data: Vec<u8>,
}

pub fn parse_module(data: &[u8]) -> Result<Module, FirmwareError> {
    use goblin::elf::Elf;
    use goblin::elf::program_header::PT_LOAD;

    let elf = Elf::parse(data)
        .map_err(|e| FirmwareError::MalformedDynlibData(format!("ELF parse: {e}")))?;

    info!(
        "SCE module: type={:#x} entry={:#x} phnum={}",
        elf.header.e_type,
        elf.header.e_entry,
        elf.program_headers.len()
    );

    let mut segments = Vec::new();
    let mut dynlib_data = Vec::new();

    for ph in &elf.program_headers {
        match ph.p_type {
            PT_LOAD => {
                let off = ph.p_offset as usize;
                let fsz = ph.p_filesz as usize;
                let mut seg = vec![0u8; ph.p_memsz as usize];
                if off + fsz <= data.len() {
                    seg[..fsz].copy_from_slice(&data[off..off + fsz]);
                }
                segments.push(ModuleSegment {
                    vaddr: ph.p_vaddr,
                    mem_size: ph.p_memsz,
                    data: seg,
                    executable: ph.p_flags & 0x1 != 0,
                });
            }
            PT_SCE_DYNLIBDATA => {
                let off = ph.p_offset as usize;
                let sz = ph.p_filesz as usize;
                if off + sz <= data.len() {
                    dynlib_data = data[off..off + sz].to_vec();
                    debug!("PT_SCE_DYNLIBDATA: offset={off:#x} size={sz:#x}");
                } else {
                    return Err(FirmwareError::MalformedDynlibData(
                        "PT_SCE_DYNLIBDATA extends beyond file".into(),
                    ));
                }
            }
            _ => {}
        }
    }

    let name = elf
        .soname
        .map(|s| s.to_string())
        .unwrap_or_else(|| "module.sprx".to_string());

    Ok(Module {
        name,
        entry_point: elf.header.e_entry,
        segments,
        dynlib_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal ELF64 with one PT_LOAD and one PT_SCE_DYNLIBDATA header.
    /// Layout: [ehdr(64)][phdr0(56)][phdr1(56)][load data][dynlib data]
    fn synthetic_sce_module(load: &[u8], dynlib: &[u8]) -> Vec<u8> {
        let ehsize = 64usize;
        let phentsize = 56usize;
        let phnum = 2usize;
        let ph_off = ehsize;
        let load_off = ph_off + phentsize * phnum;
        let dynlib_off = load_off + load.len();
        let total = dynlib_off + dynlib.len();

        let mut b = vec![0u8; total];
        // ELF header
        b[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        b[4] = 2; // 64-bit
        b[5] = 1; // little-endian
        b[6] = 1;
        b[16..18].copy_from_slice(&0xFE10u16.to_le_bytes()); // ET_SCE_DYNEXEC
        b[18..20].copy_from_slice(&0x3Eu16.to_le_bytes()); // x86-64
        b[24..32].copy_from_slice(&0x4000u64.to_le_bytes()); // e_entry
        b[32..40].copy_from_slice(&(ph_off as u64).to_le_bytes()); // e_phoff
        b[52..54].copy_from_slice(&(ehsize as u16).to_le_bytes());
        b[54..56].copy_from_slice(&(phentsize as u16).to_le_bytes());
        b[56..58].copy_from_slice(&(phnum as u16).to_le_bytes());

        // phdr0: PT_LOAD, executable
        let p0 = ph_off;
        b[p0..p0 + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
        b[p0 + 4..p0 + 8].copy_from_slice(&0x1u32.to_le_bytes()); // flags = X
        b[p0 + 8..p0 + 16].copy_from_slice(&(load_off as u64).to_le_bytes()); // p_offset
        b[p0 + 16..p0 + 24].copy_from_slice(&0x4000u64.to_le_bytes()); // p_vaddr
        b[p0 + 32..p0 + 40].copy_from_slice(&(load.len() as u64).to_le_bytes()); // p_filesz
        b[p0 + 40..p0 + 48].copy_from_slice(&(load.len() as u64).to_le_bytes()); // p_memsz

        // phdr1: PT_SCE_DYNLIBDATA
        let p1 = ph_off + phentsize;
        b[p1..p1 + 4].copy_from_slice(&PT_SCE_DYNLIBDATA.to_le_bytes());
        b[p1 + 8..p1 + 16].copy_from_slice(&(dynlib_off as u64).to_le_bytes()); // p_offset
        b[p1 + 32..p1 + 40].copy_from_slice(&(dynlib.len() as u64).to_le_bytes()); // p_filesz

        b[load_off..load_off + load.len()].copy_from_slice(load);
        b[dynlib_off..dynlib_off + dynlib.len()].copy_from_slice(dynlib);
        b
    }

    #[test]
    fn extracts_segments_and_dynlibdata() {
        let load = [0x90u8; 32]; // NOPs
        let dynlib = [0xAAu8; 48];
        let bytes = synthetic_sce_module(&load, &dynlib);
        let m = parse_module(&bytes).unwrap();
        assert_eq!(m.entry_point, 0x4000);
        assert_eq!(m.segments.len(), 1);
        assert!(m.segments[0].executable);
        assert_eq!(&m.segments[0].data[..32], &load);
        assert_eq!(m.dynlib_data, dynlib);
    }

    #[test]
    fn rejects_non_elf() {
        let err = parse_module(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }
}
```

- [ ] **Step 2: Declare the module**

In `crates/raeen-firmware/src/lib.rs`, add:
```rust
pub mod sprx;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p raeen-firmware sprx`
Expected: `extracts_segments_and_dynlibdata` and `rejects_non_elf` pass.

- [ ] **Step 4: Commit**

```bash
git add crates/raeen-firmware/src/sprx.rs crates/raeen-firmware/src/lib.rs
git commit -m "$(printf 'feat(firmware): add SCE module parser\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 7: NID hashing + `NidDatabase` (`dynlib/nid.rs`)

**Files:**
- Create: `crates/raeen-firmware/src/dynlib/mod.rs` (module root; grown in Task 8)
- Create: `crates/raeen-firmware/src/dynlib/nid.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod dynlib;`)

**Interfaces:**
- Consumes: `sha1`, `base64`.
- Produces:
  - `pub fn nid_of(name: &str) -> String` — Sony symbol NID (11-char custom-base64).
  - `pub struct NidDatabase { map: HashMap<String, String> }` with `new()`, `insert_name(&mut self, name: &str)`, `name_of(&self, nid: &str) -> Option<&str>`, `len(&self)`.

> **Validation note:** `nid_of` follows the documented SHA-1 + fixed-suffix + custom-base64 scheme. The exact byte order / alphabet must match Sony's for **real** modules; validate against a known public symbol/NID pair before relying on it at LM2+. LM1 uses homebrew where our hashing is applied consistently on both sides, so the pipeline is exercised regardless. Tests below assert determinism and round-trip, not an external vector.

- [ ] **Step 1: Write the failing tests**

Create `crates/raeen-firmware/src/dynlib/nid.rs`:
```rust
//! NID (symbol hash) computation and a NID→name database.
//!
//! Sony replaces symbol names in import/export tables with NIDs: the first
//! 8 bytes of `SHA-1(name + fixed_suffix)`, encoded with a custom base64
//! alphabet (11 chars).

use std::collections::HashMap;

use base64::Engine;
use sha1::{Digest, Sha1};

/// Fixed 16-byte suffix appended before hashing (documented Sony symbol salt).
/// Validate against a public symbol/NID pair before trusting for real modules.
const NID_SUFFIX: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Custom base64 alphabet used by Sony NIDs (`+` and `-` as the last two).
const NID_ALPHABET: &str =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

/// Compute the NID for a symbol name.
pub fn nid_of(name: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(name.as_bytes());
    hasher.update(NID_SUFFIX);
    let digest = hasher.finalize();
    let first8 = &digest[..8];

    let alphabet = base64::alphabet::Alphabet::new(NID_ALPHABET).expect("valid alphabet");
    let engine = base64::engine::GeneralPurpose::new(
        &alphabet,
        base64::engine::GeneralPurposeConfig::new()
            .with_encode_padding(false),
    );
    engine.encode(first8)
}

/// Maps NIDs back to the symbol names Raeen knows.
#[derive(Debug, Default)]
pub struct NidDatabase {
    map: HashMap<String, String>,
}

impl NidDatabase {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Register a name so its NID resolves back to it.
    pub fn insert_name(&mut self, name: &str) {
        self.map.insert(nid_of(name), name.to_string());
    }

    pub fn name_of(&self, nid: &str) -> Option<&str> {
        self.map.get(nid).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nid_is_deterministic() {
        assert_eq!(nid_of("sceKernelAllocateDirectMemory"), nid_of("sceKernelAllocateDirectMemory"));
    }

    #[test]
    fn distinct_names_differ() {
        assert_ne!(nid_of("foo"), nid_of("bar"));
    }

    #[test]
    fn nid_is_eleven_chars() {
        // 8 bytes -> 11 base64 chars (unpadded).
        assert_eq!(nid_of("module_start").len(), 11);
    }

    #[test]
    fn database_round_trips_name() {
        let mut db = NidDatabase::new();
        db.insert_name("sceKernelUsleep");
        let nid = nid_of("sceKernelUsleep");
        assert_eq!(db.name_of(&nid), Some("sceKernelUsleep"));
        assert_eq!(db.name_of("notarealnid"), None);
    }
}
```

- [ ] **Step 2: Create the `dynlib` module root**

Create `crates/raeen-firmware/src/dynlib/mod.rs`:
```rust
//! Sony dynamic-linking data: NID hashing, PT_SCE_DYNLIBDATA decoding, and
//! the NID-based linker.

pub mod nid;
```

- [ ] **Step 3: Declare the module in the crate root**

In `crates/raeen-firmware/src/lib.rs`, add:
```rust
pub mod dynlib;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p raeen-firmware nid`
Expected: 4 `dynlib::nid::tests::*` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/raeen-firmware/src/dynlib/ crates/raeen-firmware/src/lib.rs
git commit -m "$(printf 'feat(firmware): add NID hashing and NID database\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 8: `PT_SCE_DYNLIBDATA` decoder (`dynlib/mod.rs`)

**Files:**
- Modify: `crates/raeen-firmware/src/dynlib/mod.rs`

**Interfaces:**
- Consumes: `FirmwareError`.
- Produces:
  - `pub struct ImportEntry { pub nid: String, pub library_id: u16, pub module_id: u16 }`
  - `pub struct ExportEntry { pub nid: String, pub vaddr: u64 }`
  - `pub struct Relocation { pub offset: u64, pub sym_index: u32, pub rel_type: u32, pub addend: i64 }`
  - `pub struct DynlibInfo { pub imports: Vec<ImportEntry>, pub exports: Vec<ExportEntry>, pub relocations: Vec<Relocation> }`
  - `pub fn parse_dynlib(blob: &[u8]) -> Result<DynlibInfo, FirmwareError>`

> **Format note:** The DT_SCE_* tag values below are the documented values; the decoder is driven by tags found in the blob. Tests construct a synthetic blob **using these same constants**, so they are correct-by-construction. Real-module coverage (more relocation types, fingerprint-symbol nuances) is extended empirically at LM2+.

- [ ] **Step 1: Write the failing tests + decoder skeleton**

Replace the contents of `crates/raeen-firmware/src/dynlib/mod.rs` with:
```rust
//! Sony dynamic-linking data: NID hashing, PT_SCE_DYNLIBDATA decoding, and
//! the NID-based linker.

pub mod nid;

use raeen_core::error::FirmwareError;

// Documented DT_SCE_* dynamic tag values. Validated against real modules at LM2+.
const DT_SCE_STRTAB: u64 = 0x6100_0035;
const DT_SCE_STRSZ: u64 = 0x6100_0037;
const DT_SCE_SYMTAB: u64 = 0x6100_0039;
const DT_SCE_SYMTABSZ: u64 = 0x6100_003F;
const DT_SCE_JMPREL: u64 = 0x6100_002D;
const DT_SCE_PLTRELSZ: u64 = 0x6100_002B;

/// A 24-byte entry: [tag: u64][value: u64][aux: u64]. `value` is an offset into
/// the string/symbol/reloc regions that follow the tag array.
const DYN_TAG_SIZE: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportEntry {
    pub nid: String,
    pub library_id: u16,
    pub module_id: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportEntry {
    pub nid: String,
    pub vaddr: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relocation {
    pub offset: u64,
    pub sym_index: u32,
    pub rel_type: u32,
    pub addend: i64,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct DynlibInfo {
    pub imports: Vec<ImportEntry>,
    pub exports: Vec<ExportEntry>,
    pub relocations: Vec<Relocation>,
}

/// Layout consumed by the decoder (also produced by the test fixture builder):
///
/// ```text
/// [ tag array: N * 24 bytes, terminated by tag=0 ]
/// [ string table: STRSZ bytes ]
/// [ symbol table: SYMTABSZ bytes; each sym = 24 bytes:
///       name_off: u32, info: u8, _pad: u8, library_id: u16,
///       module_id: u16, is_export: u8, _pad2: u8, value: u64 ]
/// [ reloc table: PLTRELSZ bytes; each rela = 24 bytes:
///       offset: u64, info: u64 (sym_index<<32 | type), addend: i64 ]
/// ```
///
/// Offsets in DT_SCE_* tags are relative to the start of `blob`.
pub fn parse_dynlib(blob: &[u8]) -> Result<DynlibInfo, FirmwareError> {
    let mut strtab_off = 0usize;
    let mut strsz = 0usize;
    let mut symtab_off = 0usize;
    let mut symtabsz = 0usize;
    let mut jmprel_off = 0usize;
    let mut pltrelsz = 0usize;

    let mut i = 0usize;
    loop {
        if i + DYN_TAG_SIZE > blob.len() {
            return Err(FirmwareError::MalformedDynlibData(
                "tag array not terminated".into(),
            ));
        }
        let tag = u64::from_le_bytes(blob[i..i + 8].try_into().unwrap());
        let value = u64::from_le_bytes(blob[i + 8..i + 16].try_into().unwrap());
        i += DYN_TAG_SIZE;
        match tag {
            0 => break,
            DT_SCE_STRTAB => strtab_off = value as usize,
            DT_SCE_STRSZ => strsz = value as usize,
            DT_SCE_SYMTAB => symtab_off = value as usize,
            DT_SCE_SYMTABSZ => symtabsz = value as usize,
            DT_SCE_JMPREL => jmprel_off = value as usize,
            DT_SCE_PLTRELSZ => pltrelsz = value as usize,
            _ => {}
        }
    }

    let read_str = |off: usize| -> Result<String, FirmwareError> {
        let base = strtab_off + off;
        if base > strtab_off + strsz || base >= blob.len() {
            return Err(FirmwareError::MalformedDynlibData("string offset OOB".into()));
        }
        let end = blob[base..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| base + p)
            .unwrap_or(blob.len());
        Ok(String::from_utf8_lossy(&blob[base..end]).into_owned())
    };

    // Parse symbols.
    let sym_size = 24usize;
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let sym_count = symtabsz / sym_size;
    for s in 0..sym_count {
        let b = symtab_off + s * sym_size;
        if b + sym_size > blob.len() {
            return Err(FirmwareError::MalformedDynlibData("symbol OOB".into()));
        }
        let name_off = u32::from_le_bytes(blob[b..b + 4].try_into().unwrap()) as usize;
        let library_id = u16::from_le_bytes(blob[b + 6..b + 8].try_into().unwrap());
        let module_id = u16::from_le_bytes(blob[b + 8..b + 10].try_into().unwrap());
        let is_export = blob[b + 10];
        let value = u64::from_le_bytes(blob[b + 16..b + 24].try_into().unwrap());
        let nid = read_str(name_off)?;
        if is_export != 0 {
            exports.push(ExportEntry { nid, vaddr: value });
        } else {
            imports.push(ImportEntry { nid, library_id, module_id });
        }
    }

    // Parse relocations.
    let rela_size = 24usize;
    let mut relocations = Vec::new();
    let rela_count = pltrelsz / rela_size;
    for r in 0..rela_count {
        let b = jmprel_off + r * rela_size;
        if b + rela_size > blob.len() {
            return Err(FirmwareError::MalformedDynlibData("reloc OOB".into()));
        }
        let offset = u64::from_le_bytes(blob[b..b + 8].try_into().unwrap());
        let info = u64::from_le_bytes(blob[b + 8..b + 16].try_into().unwrap());
        let addend = i64::from_le_bytes(blob[b + 16..b + 24].try_into().unwrap());
        relocations.push(Relocation {
            offset,
            sym_index: (info >> 32) as u32,
            rel_type: (info & 0xffff_ffff) as u32,
            addend,
        });
    }

    Ok(DynlibInfo { imports, exports, relocations })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sym {
        name: &'static str,
        library_id: u16,
        module_id: u16,
        is_export: bool,
        value: u64,
    }

    fn build_blob(syms: &[Sym], relocs: &[Relocation]) -> Vec<u8> {
        // Build string table.
        let mut strtab = vec![0u8]; // index 0 = empty string
        let mut name_offs = Vec::new();
        for s in syms {
            name_offs.push(strtab.len() as u32);
            strtab.extend_from_slice(s.name.as_bytes());
            strtab.push(0);
        }
        // Tag array: 6 tags + terminator = 7 * 24.
        let tag_area = 7 * DYN_TAG_SIZE;
        let strtab_off = tag_area;
        let symtab_off = strtab_off + strtab.len();
        let symtabsz = syms.len() * 24;
        let jmprel_off = symtab_off + symtabsz;
        let pltrelsz = relocs.len() * 24;
        let total = jmprel_off + pltrelsz;

        let mut b = vec![0u8; total];
        let mut put_tag = |idx: usize, tag: u64, val: u64| {
            let o = idx * DYN_TAG_SIZE;
            b[o..o + 8].copy_from_slice(&tag.to_le_bytes());
            b[o + 8..o + 16].copy_from_slice(&val.to_le_bytes());
        };
        put_tag(0, DT_SCE_STRTAB, strtab_off as u64);
        put_tag(1, DT_SCE_STRSZ, strtab.len() as u64);
        put_tag(2, DT_SCE_SYMTAB, symtab_off as u64);
        put_tag(3, DT_SCE_SYMTABSZ, symtabsz as u64);
        put_tag(4, DT_SCE_JMPREL, jmprel_off as u64);
        put_tag(5, DT_SCE_PLTRELSZ, pltrelsz as u64);
        // index 6 left as terminator (tag=0).

        b[strtab_off..strtab_off + strtab.len()].copy_from_slice(&strtab);

        for (i, s) in syms.iter().enumerate() {
            let o = symtab_off + i * 24;
            b[o..o + 4].copy_from_slice(&name_offs[i].to_le_bytes());
            b[o + 6..o + 8].copy_from_slice(&s.library_id.to_le_bytes());
            b[o + 8..o + 10].copy_from_slice(&s.module_id.to_le_bytes());
            b[o + 10] = s.is_export as u8;
            b[o + 16..o + 24].copy_from_slice(&s.value.to_le_bytes());
        }

        for (i, r) in relocs.iter().enumerate() {
            let o = jmprel_off + i * 24;
            let info = ((r.sym_index as u64) << 32) | (r.rel_type as u64);
            b[o..o + 8].copy_from_slice(&r.offset.to_le_bytes());
            b[o + 8..o + 16].copy_from_slice(&info.to_le_bytes());
            b[o + 16..o + 24].copy_from_slice(&r.addend.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_imports_exports_and_relocs() {
        let syms = [
            Sym { name: "importedFn", library_id: 3, module_id: 1, is_export: false, value: 0 },
            Sym { name: "exportedFn", library_id: 0, module_id: 0, is_export: true, value: 0x5000 },
        ];
        let relocs = [Relocation { offset: 0x8000, sym_index: 0, rel_type: 7, addend: 0 }];
        let blob = build_blob(&syms, &relocs);

        let info = parse_dynlib(&blob).unwrap();
        assert_eq!(info.imports.len(), 1);
        assert_eq!(info.imports[0].nid, "importedFn");
        assert_eq!(info.imports[0].library_id, 3);
        assert_eq!(info.exports.len(), 1);
        assert_eq!(info.exports[0].nid, "exportedFn");
        assert_eq!(info.exports[0].vaddr, 0x5000);
        assert_eq!(info.relocations.len(), 1);
        assert_eq!(info.relocations[0].offset, 0x8000);
        assert_eq!(info.relocations[0].sym_index, 0);
    }

    #[test]
    fn rejects_unterminated_tag_array() {
        let b = vec![0u8; 8]; // too short for a full tag
        assert!(matches!(
            parse_dynlib(&b),
            Err(FirmwareError::MalformedDynlibData(_))
        ));
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p raeen-firmware dynlib::tests`
Expected: `parses_imports_exports_and_relocs` and `rejects_unterminated_tag_array` pass.

- [ ] **Step 3: Commit**

```bash
git add crates/raeen-firmware/src/dynlib/mod.rs
git commit -m "$(printf 'feat(firmware): decode PT_SCE_DYNLIBDATA imports/exports/relocs\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 9: `ModuleRegistry` + NID linker

**Files:**
- Modify: `crates/raeen-hle/src/lib.rs` (add `HleRegistry::entries()`)
- Create: `crates/raeen-firmware/src/registry.rs`
- Create: `crates/raeen-firmware/src/dynlib/linker.rs`
- Modify: `crates/raeen-firmware/src/dynlib/mod.rs` (add `pub mod linker;`)
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod registry;`)

**Interfaces:**
- Consumes: `raeen_hle::{HleRegistry, HleFunction}`, `crate::dynlib::{DynlibInfo, ImportEntry, Relocation, nid::NidDatabase}`, `crate::sprx::Module`.
- Produces:
  - `raeen_hle::HleRegistry::entries(&self) -> Vec<(String, String, HleFunction)>` → `(library, function, func)`.
  - `pub enum Resolver { Hle(HleFunction), Lle(u64), Unresolved }`
  - `pub enum Policy { PreferHle, PreferLle }`
  - `pub struct ModuleRegistry { … }` with:
    - `ModuleRegistry::from_hle(hle: &HleRegistry) -> ModuleRegistry`
    - `set_policy(&mut self, module: &str, policy: Policy)`
    - `add_export(&mut self, nid: &str, vaddr: u64)`
    - `resolve(&self, import: &ImportEntry) -> Resolver`
  - `pub struct LinkedModule { pub name: String, pub entry_point: u64, pub segments: Vec<crate::sprx::ModuleSegment>, pub unresolved: Vec<String> }`
  - `pub fn link_module(module: crate::sprx::Module, dynlib: &DynlibInfo, registry: &ModuleRegistry) -> Result<LinkedModule, FirmwareError>`

- [ ] **Step 1: Add `entries()` to `HleRegistry`**

In `crates/raeen-hle/src/lib.rs`, add this method inside `impl HleRegistry` (after `is_implemented`):
```rust
    /// Enumerate all registered functions as `(library, function, impl)`.
    /// Used by the firmware module registry to build NID→HLE bindings.
    pub fn entries(&self) -> Vec<(String, String, HleFunction)> {
        self.functions
            .iter()
            .filter_map(|kv| {
                let key = kv.key();
                let (library, function) = key.split_once("::")?;
                Some((library.to_string(), function.to_string(), *kv.value()))
            })
            .collect()
    }
```

- [ ] **Step 2: Write the failing tests + `registry.rs`**

Create `crates/raeen-firmware/src/registry.rs`:
```rust
//! Per-module HLE/LLE dispatch. Each import NID resolves to an HLE function,
//! an export of another loaded module (LLE), or is left unresolved.

use std::collections::HashMap;

use tracing::debug;
use raeen_hle::{HleFunction, HleRegistry};

use crate::dynlib::ImportEntry;
use crate::dynlib::nid::nid_of;

#[derive(Clone, Copy)]
pub enum Resolver {
    Hle(HleFunction),
    Lle(u64),
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    PreferHle,
    PreferLle,
}

pub struct ModuleRegistry {
    /// NID → HLE implementation (bridged from `raeen-hle` by hashing names).
    hle_by_nid: HashMap<String, HleFunction>,
    /// NID → LLE export address (from other loaded modules).
    lle_by_nid: HashMap<String, u64>,
    /// Per-module resolution preference.
    policy: HashMap<String, Policy>,
}

impl ModuleRegistry {
    /// Build a registry that bridges every HLE function to its NID.
    pub fn from_hle(hle: &HleRegistry) -> Self {
        let mut hle_by_nid = HashMap::new();
        for (_library, function, f) in hle.entries() {
            hle_by_nid.insert(nid_of(&function), f);
        }
        debug!("ModuleRegistry: bridged {} HLE functions", hle_by_nid.len());
        Self {
            hle_by_nid,
            lle_by_nid: HashMap::new(),
            policy: HashMap::new(),
        }
    }

    pub fn set_policy(&mut self, module: &str, policy: Policy) {
        self.policy.insert(module.to_string(), policy);
    }

    /// Register an export from a loaded real module (LLE side).
    pub fn add_export(&mut self, nid: &str, vaddr: u64) {
        self.lle_by_nid.insert(nid.to_string(), vaddr);
    }

    pub fn resolve(&self, import: &ImportEntry) -> Resolver {
        let prefer_lle = matches!(
            self.policy.get(&import.module_id.to_string()),
            Some(Policy::PreferLle)
        );
        let hle = self.hle_by_nid.get(&import.nid).copied();
        let lle = self.lle_by_nid.get(&import.nid).copied();

        match (prefer_lle, hle, lle) {
            (true, _, Some(a)) => Resolver::Lle(a),
            (true, Some(f), None) => Resolver::Hle(f),
            (false, Some(f), _) => Resolver::Hle(f),
            (false, None, Some(a)) => Resolver::Lle(a),
            _ => Resolver::Unresolved,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::nid::nid_of;

    fn stub_impl(_args: &[u64]) -> u64 {
        0x1234
    }

    #[test]
    fn resolves_import_to_hle() {
        let hle = HleRegistry::new();
        hle.register("libkernel", "sceKernelUsleep", stub_impl);
        let reg = ModuleRegistry::from_hle(&hle);

        let import = ImportEntry {
            nid: nid_of("sceKernelUsleep"),
            library_id: 0,
            module_id: 0,
        };
        match reg.resolve(&import) {
            Resolver::Hle(f) => assert_eq!(f(&[]), 0x1234),
            _ => panic!("expected HLE resolution"),
        }
    }

    #[test]
    fn resolves_export_to_lle_when_no_hle() {
        let hle = HleRegistry::new();
        let mut reg = ModuleRegistry::from_hle(&hle);
        reg.add_export(&nid_of("customExport"), 0x9000);

        let import = ImportEntry {
            nid: nid_of("customExport"),
            library_id: 0,
            module_id: 0,
        };
        assert!(matches!(reg.resolve(&import), Resolver::Lle(0x9000)));
    }

    #[test]
    fn unknown_nid_is_unresolved() {
        let hle = HleRegistry::new();
        let reg = ModuleRegistry::from_hle(&hle);
        let import = ImportEntry {
            nid: nid_of("neverRegistered"),
            library_id: 0,
            module_id: 0,
        };
        assert!(matches!(reg.resolve(&import), Resolver::Unresolved));
    }
}
```

- [ ] **Step 3: Write `dynlib/linker.rs`**

Create `crates/raeen-firmware/src/dynlib/linker.rs`:
```rust
//! NID-based linker: resolve each import via the registry and patch the
//! module's segments with the resolved addresses.

use tracing::{debug, warn};
use raeen_core::error::FirmwareError;

use crate::dynlib::DynlibInfo;
use crate::registry::{ModuleRegistry, Resolver};
use crate::sprx::{Module, ModuleSegment};

/// A module whose imports have been resolved and relocations applied.
#[derive(Debug)]
pub struct LinkedModule {
    pub name: String,
    pub entry_point: u64,
    pub segments: Vec<ModuleSegment>,
    /// NIDs that could not be resolved (logged, non-fatal).
    pub unresolved: Vec<String>,
}

/// Synthetic address handed to unresolved imports so execution can trap
/// deliberately rather than jump to zero.
const UNRESOLVED_STUB_ADDR: u64 = 0xDEAD_0000_0000_0000;

/// Write an 8-byte little-endian value at `vaddr` within whichever segment
/// contains it. Returns false if no segment covers the address.
fn patch_u64(segments: &mut [ModuleSegment], vaddr: u64, value: u64) -> bool {
    for seg in segments.iter_mut() {
        if vaddr >= seg.vaddr && vaddr + 8 <= seg.vaddr + seg.data.len() as u64 {
            let off = (vaddr - seg.vaddr) as usize;
            seg.data[off..off + 8].copy_from_slice(&value.to_le_bytes());
            return true;
        }
    }
    false
}

pub fn link_module(
    module: Module,
    dynlib: &DynlibInfo,
    registry: &ModuleRegistry,
) -> Result<LinkedModule, FirmwareError> {
    let mut segments = module.segments;
    let mut unresolved = Vec::new();

    for reloc in &dynlib.relocations {
        let import = match dynlib.imports.get(reloc.sym_index as usize) {
            Some(i) => i,
            None => {
                return Err(FirmwareError::MalformedDynlibData(format!(
                    "relocation references sym_index {} out of {} imports",
                    reloc.sym_index,
                    dynlib.imports.len()
                )));
            }
        };
        let addr = match registry.resolve(import) {
            Resolver::Hle(_) => {
                // HLE functions are dispatched by NID at call time; the patched
                // value is a tagged sentinel the runtime recognizes. For the
                // spine we mark it resolved by writing a deterministic tag
                // (0x41E0 high bits = "HLE-resolved") that stays distinguishable
                // per import.
                0x41E0_0000_0000_0000u64 ^ fnv1a(&import.nid)
            }
            Resolver::Lle(a) => a.wrapping_add(reloc.addend as u64),
            Resolver::Unresolved => {
                warn!("unresolved import NID {}", import.nid);
                unresolved.push(import.nid.clone());
                UNRESOLVED_STUB_ADDR
            }
        };
        if !patch_u64(&mut segments, reloc.offset, addr) {
            return Err(FirmwareError::MalformedDynlibData(format!(
                "relocation offset {:#x} not in any segment",
                reloc.offset
            )));
        }
        debug!("patched {:#x} <- {:#x} ({})", reloc.offset, addr, import.nid);
    }

    Ok(LinkedModule {
        name: module.name,
        entry_point: module.entry_point,
        segments,
        unresolved,
    })
}

/// Small deterministic hash used to make HLE-resolved slots distinguishable.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h & 0x0000_FFFF_FFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::{ImportEntry, Relocation};
    use crate::registry::ModuleRegistry;
    use crate::sprx::ModuleSegment;
    use raeen_hle::HleRegistry;

    fn stub(_a: &[u64]) -> u64 {
        7
    }

    fn one_segment_module() -> Module {
        Module {
            name: "test.sprx".into(),
            entry_point: 0x4000,
            segments: vec![ModuleSegment {
                vaddr: 0x4000,
                mem_size: 0x100,
                data: vec![0u8; 0x100],
                executable: true,
            }],
            dynlib_data: vec![],
        }
    }

    #[test]
    fn applies_relocation_for_resolved_import() {
        let hle = HleRegistry::new();
        hle.register("libkernel", "sceFoo", stub);
        let registry = ModuleRegistry::from_hle(&hle);

        let dynlib = DynlibInfo {
            imports: vec![ImportEntry {
                nid: crate::dynlib::nid::nid_of("sceFoo"),
                library_id: 0,
                module_id: 0,
            }],
            exports: vec![],
            relocations: vec![Relocation {
                offset: 0x4010,
                sym_index: 0,
                rel_type: 7,
                addend: 0,
            }],
        };

        let linked = link_module(one_segment_module(), &dynlib, &registry).unwrap();
        assert!(linked.unresolved.is_empty());
        // The slot at 0x4010 (offset 0x10 in the segment) was overwritten.
        assert_ne!(&linked.segments[0].data[0x10..0x18], &[0u8; 8]);
    }

    #[test]
    fn records_unresolved_imports() {
        let hle = HleRegistry::new();
        let registry = ModuleRegistry::from_hle(&hle);
        let dynlib = DynlibInfo {
            imports: vec![ImportEntry {
                nid: crate::dynlib::nid::nid_of("sceMissing"),
                library_id: 0,
                module_id: 0,
            }],
            exports: vec![],
            relocations: vec![Relocation {
                offset: 0x4020,
                sym_index: 0,
                rel_type: 7,
                addend: 0,
            }],
        };
        let linked = link_module(one_segment_module(), &dynlib, &registry).unwrap();
        assert_eq!(linked.unresolved, vec![crate::dynlib::nid::nid_of("sceMissing")]);
    }
}
```

- [ ] **Step 4: Wire the new modules**

In `crates/raeen-firmware/src/dynlib/mod.rs`, add after `pub mod nid;`:
```rust
pub mod linker;
```
In `crates/raeen-firmware/src/lib.rs`, add:
```rust
pub mod registry;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p raeen-firmware -p raeen-hle`
Expected: registry (3) + linker (2) tests pass; `raeen-hle` still green.

- [ ] **Step 6: Commit**

```bash
git add crates/raeen-hle/src/lib.rs crates/raeen-firmware/src/registry.rs crates/raeen-firmware/src/dynlib/
git commit -m "$(printf 'feat(firmware): add module registry and NID linker\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 10: `ModuleLoader` orchestration + LM1 end-to-end test

**Files:**
- Create: `crates/raeen-firmware/src/loader.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod loader;` + re-exports)
- Create: `crates/raeen-firmware/tests/lm1_end_to_end.rs`

**Interfaces:**
- Consumes: all prior tasks.
- Produces:
  - `pub struct ModuleLoader<'a> { provider: &'a dyn KeyProvider, registry: &'a ModuleRegistry }`
  - `ModuleLoader::new(provider, registry)`
  - `ModuleLoader::load_module(&self, bytes: &[u8]) -> Result<LinkedModule, FirmwareError>` — SELF-detect → decrypt-or-passthrough → SCE parse → dynlib decode → link.

- [ ] **Step 1: Write `loader.rs`**

Create `crates/raeen-firmware/src/loader.rs`:
```rust
//! Top-level module ingestion: bytes → linked module.

use tracing::info;
use raeen_core::crypto::KeyProvider;
use raeen_core::error::FirmwareError;

use crate::dynlib::linker::{link_module, LinkedModule};
use crate::dynlib::parse_dynlib;
use crate::registry::ModuleRegistry;
use crate::sprx::parse_module;

/// SELF outer magic (mirrors `raeen_loader::self_format`).
const SELF_MAGIC: u32 = 0x4F15_D17E;

pub struct ModuleLoader<'a> {
    provider: &'a dyn KeyProvider,
    registry: &'a ModuleRegistry,
}

impl<'a> ModuleLoader<'a> {
    pub fn new(provider: &'a dyn KeyProvider, registry: &'a ModuleRegistry) -> Self {
        Self { provider, registry }
    }

    /// Load a module from raw bytes (a SELF-wrapped or bare SCE ELF).
    pub fn load_module(&self, bytes: &[u8]) -> Result<LinkedModule, FirmwareError> {
        let elf_bytes: std::borrow::Cow<[u8]> = if is_self(bytes) {
            info!("input is SELF; running through decryption seam");
            // Decrypt (or pass through) via the loader's provider-aware path,
            // then re-serialize is unnecessary: parse_self_with_provider returns
            // a LoadedBinary, but we need the raw ELF for SCE dynlib extraction.
            // For the spine we require decrypted-or-homebrew input: extract the
            // inner ELF region directly, decrypting if a key is available.
            std::borrow::Cow::Owned(decrypt_self_inner(bytes, self.provider)?)
        } else {
            std::borrow::Cow::Borrowed(bytes)
        };

        let module = parse_module(&elf_bytes)?;
        let dynlib = if module.dynlib_data.is_empty() {
            Default::default()
        } else {
            parse_dynlib(&module.dynlib_data)?
        };
        link_module(module, &dynlib, self.registry)
    }
}

fn is_self(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == SELF_MAGIC
}

/// Extract the inner ELF from a SELF, decrypting via the provider if needed.
/// Mirrors the field layout used by `raeen_loader::self_format`.
fn decrypt_self_inner(
    data: &[u8],
    provider: &dyn KeyProvider,
) -> Result<Vec<u8>, FirmwareError> {
    use raeen_core::crypto::KeyRequest;

    if data.len() < 32 {
        return Err(FirmwareError::MalformedDynlibData("SELF too small".into()));
    }
    let key_type = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let header_size = u16::from_le_bytes(data[12..14].try_into().unwrap()) as usize;
    let num_entries = u16::from_le_bytes(data[24..26].try_into().unwrap()) as usize;
    if header_size >= data.len() {
        return Err(FirmwareError::MalformedDynlibData("SELF header_size OOB".into()));
    }

    let mut encrypted = false;
    for i in 0..num_entries {
        let base = 32 + i * 32;
        if base + 32 > data.len() {
            break;
        }
        let properties = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let usz = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());
        let is_enc = (properties >> 1) & 0x7 != 0;
        if is_enc && usz > 0 {
            encrypted = true;
        }
    }

    let inner = &data[header_size..];
    if !encrypted {
        return Ok(inner.to_vec());
    }
    let req = KeyRequest { key_type, key_id: 0, segment_index: 0 };
    match provider.segment_key(&req) {
        Some(k) => Ok(decrypt_cbc(inner, &k.key, &k.iv)?),
        None => Err(FirmwareError::MissingKey { key_id: 0 }),
    }
}

fn decrypt_cbc(data: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Result<Vec<u8>, FirmwareError> {
    use aes::cipher::{BlockDecryptMut, KeyIvInit};
    type Dec = cbc::Decryptor<aes::Aes128>;
    if data.len() % 16 != 0 {
        return Err(FirmwareError::MalformedDynlibData(
            "encrypted region not block-aligned".into(),
        ));
    }
    let mut out = data.to_vec();
    let mut dec = Dec::new(key.into(), iv.into());
    for chunk in out.chunks_exact_mut(16) {
        let block = aes::cipher::generic_array::GenericArray::from_mut_slice(chunk);
        dec.decrypt_block_mut(block);
    }
    Ok(out)
}
```

- [ ] **Step 2: Add `aes`/`cbc` to the firmware crate manifest**

`loader.rs` uses `aes`/`cbc`. In `crates/raeen-firmware/Cargo.toml` `[dependencies]`, add:
```toml
aes = { workspace = true }
cbc = { workspace = true }
```

- [ ] **Step 3: Declare and re-export**

In `crates/raeen-firmware/src/lib.rs`, add:
```rust
pub mod loader;

pub use dynlib::linker::LinkedModule;
pub use loader::ModuleLoader;
pub use registry::{ModuleRegistry, Policy};
```

- [ ] **Step 4: Write the LM1 end-to-end test**

Create `crates/raeen-firmware/tests/lm1_end_to_end.rs`:
```rust
//! LM1 acceptance: a homebrew SCE module flows through the whole spine —
//! SELF passthrough → SCE parse → dynlib decode → NID link against HLE stubs.

use raeen_core::crypto::NoKeysProvider;
use raeen_firmware::dynlib::nid::nid_of;
use raeen_firmware::{ModuleLoader, ModuleRegistry};
use raeen_hle::HleRegistry;

// --- fixture builders (mirror the per-module test builders) ---

const DYN_TAG_SIZE: usize = 24;
const DT_SCE_STRTAB: u64 = 0x6100_0035;
const DT_SCE_STRSZ: u64 = 0x6100_0037;
const DT_SCE_SYMTAB: u64 = 0x6100_0039;
const DT_SCE_SYMTABSZ: u64 = 0x6100_003F;
const DT_SCE_JMPREL: u64 = 0x6100_002D;
const DT_SCE_PLTRELSZ: u64 = 0x6100_002B;
const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;

fn build_dynlib(import_name: &str, reloc_offset: u64) -> Vec<u8> {
    // string table: [0][import_name\0]
    let mut strtab = vec![0u8];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(import_name.as_bytes());
    strtab.push(0);

    let tag_area = 7 * DYN_TAG_SIZE;
    let strtab_off = tag_area;
    let symtab_off = strtab_off + strtab.len();
    let symtabsz = 24; // one import symbol
    let jmprel_off = symtab_off + symtabsz;
    let pltrelsz = 24; // one relocation
    let total = jmprel_off + pltrelsz;

    let mut b = vec![0u8; total];
    let mut put = |i: usize, tag: u64, val: u64| {
        let o = i * DYN_TAG_SIZE;
        b[o..o + 8].copy_from_slice(&tag.to_le_bytes());
        b[o + 8..o + 16].copy_from_slice(&val.to_le_bytes());
    };
    put(0, DT_SCE_STRTAB, strtab_off as u64);
    put(1, DT_SCE_STRSZ, strtab.len() as u64);
    put(2, DT_SCE_SYMTAB, symtab_off as u64);
    put(3, DT_SCE_SYMTABSZ, symtabsz as u64);
    put(4, DT_SCE_JMPREL, jmprel_off as u64);
    put(5, DT_SCE_PLTRELSZ, pltrelsz as u64);

    b[strtab_off..strtab_off + strtab.len()].copy_from_slice(&strtab);
    // import symbol (is_export = 0)
    b[symtab_off..symtab_off + 4].copy_from_slice(&name_off.to_le_bytes());
    // reloc: offset, info(sym_index=0,type=7), addend=0
    let info: u64 = 7;
    b[jmprel_off..jmprel_off + 8].copy_from_slice(&reloc_offset.to_le_bytes());
    b[jmprel_off + 8..jmprel_off + 16].copy_from_slice(&info.to_le_bytes());
    b
}

fn build_sce_elf(load: &[u8], dynlib: &[u8], entry: u64, load_vaddr: u64) -> Vec<u8> {
    let ehsize = 64usize;
    let phentsize = 56usize;
    let phnum = 2usize;
    let ph_off = ehsize;
    let load_off = ph_off + phentsize * phnum;
    let dynlib_off = load_off + load.len();
    let total = dynlib_off + dynlib.len();

    let mut b = vec![0u8; total];
    b[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
    b[4] = 2;
    b[5] = 1;
    b[6] = 1;
    b[16..18].copy_from_slice(&0xFE10u16.to_le_bytes()); // ET_SCE_DYNEXEC
    b[18..20].copy_from_slice(&0x3Eu16.to_le_bytes());
    b[24..32].copy_from_slice(&entry.to_le_bytes());
    b[32..40].copy_from_slice(&(ph_off as u64).to_le_bytes());
    b[52..54].copy_from_slice(&(ehsize as u16).to_le_bytes());
    b[54..56].copy_from_slice(&(phentsize as u16).to_le_bytes());
    b[56..58].copy_from_slice(&(phnum as u16).to_le_bytes());

    let p0 = ph_off;
    b[p0..p0 + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    b[p0 + 4..p0 + 8].copy_from_slice(&0x5u32.to_le_bytes()); // R+X
    b[p0 + 8..p0 + 16].copy_from_slice(&(load_off as u64).to_le_bytes());
    b[p0 + 16..p0 + 24].copy_from_slice(&load_vaddr.to_le_bytes());
    b[p0 + 32..p0 + 40].copy_from_slice(&(load.len() as u64).to_le_bytes());
    b[p0 + 40..p0 + 48].copy_from_slice(&(load.len() as u64).to_le_bytes());

    let p1 = ph_off + phentsize;
    b[p1..p1 + 4].copy_from_slice(&PT_SCE_DYNLIBDATA.to_le_bytes());
    b[p1 + 8..p1 + 16].copy_from_slice(&(dynlib_off as u64).to_le_bytes());
    b[p1 + 32..p1 + 40].copy_from_slice(&(dynlib.len() as u64).to_le_bytes());

    b[load_off..load_off + load.len()].copy_from_slice(load);
    b[dynlib_off..dynlib_off + dynlib.len()].copy_from_slice(dynlib);
    b
}

fn hle_stub(_a: &[u64]) -> u64 {
    0x42
}

#[test]
fn homebrew_module_links_against_hle_stubs() {
    // HLE registry exposes sceKernelUsleep.
    let hle = HleRegistry::new();
    hle.register("libkernel", "sceKernelUsleep", hle_stub);
    let registry = ModuleRegistry::from_hle(&hle);

    // A homebrew module that imports sceKernelUsleep, with a GOT slot at 0x4010.
    let load_vaddr = 0x4000u64;
    let dynlib = build_dynlib("sceKernelUsleep", 0x4010);
    let elf = build_sce_elf(&[0x90u8; 0x40], &dynlib, load_vaddr, load_vaddr);

    let provider = NoKeysProvider;
    let loader = ModuleLoader::new(&provider, &registry);
    let linked = loader.load_module(&elf).unwrap();

    assert_eq!(linked.entry_point, load_vaddr);
    assert!(linked.unresolved.is_empty(), "import should resolve to HLE");
    // GOT slot at vaddr 0x4010 → offset 0x10 in the segment; it was patched.
    assert_ne!(&linked.segments[0].data[0x10..0x18], &[0u8; 8]);
}

#[test]
fn unknown_import_is_reported_not_fatal() {
    let hle = HleRegistry::new();
    let registry = ModuleRegistry::from_hle(&hle);

    let load_vaddr = 0x4000u64;
    let dynlib = build_dynlib("sceNotImplemented", 0x4010);
    let elf = build_sce_elf(&[0u8; 0x40], &dynlib, load_vaddr, load_vaddr);

    let provider = NoKeysProvider;
    let loader = ModuleLoader::new(&provider, &registry);
    let linked = loader.load_module(&elf).unwrap();

    assert_eq!(linked.unresolved, vec![nid_of("sceNotImplemented")]);
}
```

- [ ] **Step 5: Run the LM1 tests to verify they pass**

Run: `cargo test -p raeen-firmware --test lm1_end_to_end`
Expected: `homebrew_module_links_against_hle_stubs` and `unknown_import_is_reported_not_fatal` pass.

- [ ] **Step 6: Full workspace green + lints**

Run: `cargo test --workspace`
Then: `cargo clippy -p raeen-firmware -- -D warnings`
Then: `cargo fmt -p raeen-firmware`
Expected: tests pass; no clippy warnings; formatting clean.

- [ ] **Step 7: Commit**

```bash
git add crates/raeen-firmware/ Cargo.lock
git commit -m "$(printf 'feat(firmware): add ModuleLoader and LM1 end-to-end pipeline\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**Spec coverage:**
- §3.1 new `raeen-firmware` crate → Task 1. ✅
- §3.3.1 `pup.rs` + `--firmware-info` → Tasks 2, 3. ✅
- §3.3.2 `KeyProvider` seam + SELF decryption → Tasks 4, 5. ✅ (trait relocated to `raeen-core` to avoid a `loader ↔ firmware` cycle; noted in File Structure.)
- §3.3.3 `sprx.rs` → Task 6. ✅
- §3.3.4 `dynlib/` NID + dynlibdata → Tasks 7, 8. ✅
- §3.3.5 `registry.rs` HLE/LLE dispatch → Task 9. ✅
- §3.4 error handling (`FirmwareError`, demoted `EncryptedSelf`) → Tasks 1, 5. ✅
- §4 LM0 acceptance → Task 3 Step 6; LM1 acceptance → Task 10 Step 5. ✅
- §6 verification (PUP, crypto seam, NID, dynlib, linker/registry tests; guardrails) → covered across tasks; clippy/fmt in Task 10 Step 6. ✅

**Type consistency check:**
- `KeyProvider::segment_key(&self, &KeyRequest) -> Option<SegmentKey>` — identical in Tasks 4, 5, 10. ✅
- `SegmentKey { key: [u8;16], iv: [u8;16] }` — consistent across Tasks 4, 5, 10. ✅
- `ImportEntry { nid: String, library_id: u16, module_id: u16 }` — Tasks 8, 9, and linker. ✅
- `HleFunction = fn(&[u64]) -> u64` — from existing `raeen-hle`; used unchanged in Tasks 9, 10. ✅
- `HleRegistry::entries() -> Vec<(String, String, HleFunction)>` — defined Task 9 Step 1, consumed Task 9 Step 2. ✅
- `ModuleRegistry::from_hle / resolve / add_export / set_policy` — defined Task 9, consumed Task 10. ✅
- `LinkedModule { name, entry_point, segments, unresolved }` — defined Task 9, re-exported/consumed Task 10. ✅

**Placeholder scan:** No `TBD`/`TODO`/"add error handling"; every code step contains complete code. The `Default::default()` for an empty `DynlibInfo` in Task 10 relies on `#[derive(Default)]` added in Task 8. ✅
