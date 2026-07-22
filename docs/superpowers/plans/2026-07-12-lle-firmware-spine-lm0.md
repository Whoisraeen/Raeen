# LLE Firmware Spine — LM0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation of the `raeen-firmware` crate — parse the SLB2/PUP firmware container, expose its entries, install the user-supplied `KeyProvider` decryption seam, provide the SCE NID hashing/encoding library, and ship an `raeen --firmware-info <PUP>` diagnostic (milestone **LM0**).

**Architecture:** A new `raeen-firmware` crate depends on `raeen-core` (errors/types) and `raeen-loader` (ELF/SELF primitives). It parses the plaintext SLB2 container structure without touching encryption, holds the `KeyProvider` trait behind which real decryption plugs in later, and computes SCE NIDs so a future linker can resolve imports. The GUI binary gains a diagnostic flag that inspects a firmware package and exits.

**Tech Stack:** Rust (edition 2024), `memmap2` (map the ~1.2 GB PUP without reading it into RAM), `sha1` (NID hashing), `tracing` (logging), `thiserror`-based errors from `raeen-core`.

## Global Constraints

- **Clean-room boundary (spec §2):** Raeen ships **no keys, no firmware, no key-extraction/circumvention tooling**. Code consumes keys through a `KeyProvider`; it never derives, guesses, brute-forces, or extracts them.
- **No real firmware bytes in tests.** All parser tests build synthetic byte buffers the test controls. The real `PS5 Firmware/PS5UPDATE.PUP` is used only in manual verification steps.
- **`PS5 Firmware/` stays git-ignored** (already in `.gitignore`); never commit firmware or key material.
- **Missing keys are non-fatal:** decryption without a key logs at `info` and returns `FirmwareError::MissingKey { key_id }`, never a panic or hard error.
- **Rust edition 2024, `rust-version` 1.85** (inherited from the workspace).
- **Every task ends clippy-clean:** `cargo clippy -p raeen-firmware --all-targets` must emit zero warnings before commit.
- **Errors** come from `raeen_core::error::FirmwareError` (variants: `InvalidPupMagic(u32)`, `PupEntryOutOfBounds { index: usize }`, `MissingKey { key_id: u64 }`, `UnsupportedRelocation(u32)`, `MalformedDynlibData(String)`, `Loader(#[from] LoaderError)`). `LoaderError` has `Io(#[from] std::io::Error)`, so `io::Error → LoaderError → FirmwareError` chains.

---

### Task 1: Scaffold the `raeen-firmware` crate

**Files:**
- Create: `crates/raeen-firmware/Cargo.toml`
- Create: `crates/raeen-firmware/src/lib.rs`
- Modify: `Cargo.toml` (workspace root — add crate to `members`, add `sha1` and the crate path to `workspace.dependencies`)

**Interfaces:**
- Produces: the `raeen-firmware` crate, buildable as a workspace member; a `raeen_firmware::CRATE_NAME: &str` constant used only to give Task 1 a trivial passing test.

- [ ] **Step 1: Create the crate manifest**

Create `crates/raeen-firmware/Cargo.toml`:

```toml
[package]
name = "raeen-firmware"
description = "PS5 firmware ingestion — PUP parsing, user-supplied key decryption seam, SCE module loading and NID linking"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
raeen-core = { workspace = true }
raeen-loader = { workspace = true }
tracing = { workspace = true }
memmap2 = { workspace = true }
sha1 = { workspace = true }
```

- [ ] **Step 2: Create the crate root with a smoke-test constant**

Create `crates/raeen-firmware/src/lib.rs`:

```rust
//! # Raeen Firmware
//!
//! The "firmware spine": ingests PS5 firmware packages (PUP/SLB2),
//! decrypts SELF/module payloads through a **user-supplied** [`KeyProvider`]
//! (Raeen ships no keys), and — in later milestones — parses and links
//! Sony's real `.sprx` modules by NID against HLE or LLE implementations.
//!
//! This crate never contains or extracts Sony keys or firmware. See the
//! design spec, section 2, for the clean-room boundary.

/// Crate identifier, used in diagnostics.
pub const CRATE_NAME: &str = "raeen-firmware";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_set() {
        assert_eq!(super::CRATE_NAME, "raeen-firmware");
    }
}
```

- [ ] **Step 3: Wire the crate into the workspace**

In the root `Cargo.toml`, add `"crates/raeen-firmware",` to the `members` list (keep it alongside the other `crates/raeen-*` entries). Then, in `[workspace.dependencies]`, add the external dep near the other hashing/parsing deps:

```toml
sha1 = "0.10"
```

and add the internal crate path alongside the other `raeen-*` path entries:

```toml
raeen-firmware = { path = "crates/raeen-firmware" }
```

- [ ] **Step 4: Build and test**

Run: `cargo test -p raeen-firmware`
Expected: compiles; `test crate_name_is_set ... ok`; `test result: ok. 1 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/raeen-firmware/Cargo.toml crates/raeen-firmware/src/lib.rs Cargo.toml Cargo.lock
git commit -m "feat(firmware): scaffold raeen-firmware crate"
```

---

### Task 2: SLB2 container parser

**Files:**
- Create: `crates/raeen-firmware/src/slb2.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod slb2;` and re-export)

**Interfaces:**
- Produces:
  - `pub struct Slb2Entry { pub name: String, pub offset: u64, pub size: u64 }` (derives `Debug, Clone, PartialEq, Eq`)
  - `pub fn parse_slb2(data: &[u8]) -> Result<Vec<Slb2Entry>, raeen_core::error::FirmwareError>`

**Format note:** SLB2 is a `0x20`-byte header (`"SLB2"` magic, `version`, `flags`, `file_count` at `0x0C`, `block_count` at `0x10`, reserved) followed by `file_count` entries of `0x30` bytes each: `block_offset` (u32, in 512-byte blocks), `size` (u32, bytes), 8 reserved bytes, then a `0x20`-byte null-terminated `name`. Payloads live at `block_offset * 512`.

- [ ] **Step 1: Write the failing tests**

Add `pub mod slb2;` to `crates/raeen-firmware/src/lib.rs` (below the `CRATE_NAME` const), then create `crates/raeen-firmware/src/slb2.rs` with **only** this test module at the bottom (the implementation comes in Step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use raeen_core::error::FirmwareError;

    /// Build a synthetic SLB2 container with one entry.
    fn synthetic_slb2() -> Vec<u8> {
        let mut buf = vec![0u8; 0x20 + 0x30];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[4..8].copy_from_slice(&3u32.to_le_bytes()); // version
        buf[0x0C..0x10].copy_from_slice(&1u32.to_le_bytes()); // file_count = 1
        // entry 0 at 0x20
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes()); // block_offset = 2
        buf[0x24..0x28].copy_from_slice(&0x100u32.to_le_bytes()); // size = 256
        let name = b"PS5UPDATE1.PUP";
        buf[0x30..0x30 + name.len()].copy_from_slice(name);
        buf
    }

    #[test]
    fn parses_single_entry() {
        let entries = parse_slb2(&synthetic_slb2()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "PS5UPDATE1.PUP");
        assert_eq!(entries[0].offset, 2 * 0x200);
        assert_eq!(entries[0].size, 0x100);
    }

    #[test]
    fn rejects_bad_magic() {
        let data = [0u8; 0x20];
        assert!(matches!(
            parse_slb2(&data),
            Err(FirmwareError::InvalidPupMagic(_))
        ));
    }

    #[test]
    fn rejects_truncated_entry_table() {
        let mut buf = synthetic_slb2();
        buf.truncate(0x20 + 0x10); // header + half an entry
        assert!(matches!(
            parse_slb2(&buf),
            Err(FirmwareError::PupEntryOutOfBounds { index: 0 })
        ));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p raeen-firmware slb2`
Expected: FAIL to compile — `cannot find function parse_slb2`.

- [ ] **Step 3: Write the implementation**

At the **top** of `crates/raeen-firmware/src/slb2.rs` (above the test module):

```rust
//! SLB2 firmware container parser.
//!
//! PS5 update packages are wrapped in a plaintext "SLB2" container: a fixed
//! `0x20`-byte header followed by a table of `0x30`-byte entries, each naming
//! an inner file (e.g. `PS5UPDATE1.PUP`) with its block offset and byte size.
//! Entry *contents* may be encrypted; this parser only reads the container
//! structure and never attempts decryption.

use raeen_core::error::FirmwareError;

const SLB2_MAGIC: [u8; 4] = *b"SLB2";
const SLB2_BLOCK_SIZE: u64 = 0x200; // 512 bytes
const SLB2_HEADER_SIZE: usize = 0x20;
const SLB2_ENTRY_SIZE: usize = 0x30;
const SLB2_NAME_OFFSET: usize = 0x10; // within an entry
const SLB2_NAME_LEN: usize = 0x20;

/// A parsed SLB2 container entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slb2Entry {
    /// Inner file name (e.g. "PS5UPDATE1.PUP").
    pub name: String,
    /// Byte offset of the entry data within the container.
    pub offset: u64,
    /// Size of the entry data in bytes.
    pub size: u64,
}

/// Parse the SLB2 header and entry table from `data`.
///
/// `data` must contain at least the header and full entry table; entry
/// payloads may lie beyond the slice and are not required here.
///
/// # Errors
///
/// - [`FirmwareError::InvalidPupMagic`] if the magic is not `"SLB2"`.
/// - [`FirmwareError::PupEntryOutOfBounds`] if an entry record is truncated.
pub fn parse_slb2(data: &[u8]) -> Result<Vec<Slb2Entry>, FirmwareError> {
    if data.len() < SLB2_HEADER_SIZE || data[0..4] != SLB2_MAGIC {
        let magic = if data.len() >= 4 {
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        } else {
            0
        };
        return Err(FirmwareError::InvalidPupMagic(magic));
    }

    let file_count = u32::from_le_bytes(data[0x0C..0x10].try_into().unwrap()) as usize;

    let mut entries = Vec::with_capacity(file_count);
    for index in 0..file_count {
        let base = SLB2_HEADER_SIZE + index * SLB2_ENTRY_SIZE;
        if base + SLB2_ENTRY_SIZE > data.len() {
            return Err(FirmwareError::PupEntryOutOfBounds { index });
        }
        let block_offset = u32::from_le_bytes(data[base..base + 4].try_into().unwrap()) as u64;
        let size = u32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()) as u64;
        let name_start = base + SLB2_NAME_OFFSET;
        let name = data[name_start..name_start + SLB2_NAME_LEN]
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect::<String>();
        entries.push(Slb2Entry {
            name,
            offset: block_offset * SLB2_BLOCK_SIZE,
            size,
        });
    }
    Ok(entries)
}
```

Then update the re-export in `crates/raeen-firmware/src/lib.rs` so the module is declared and its type surfaced:

```rust
pub mod slb2;

pub use slb2::{parse_slb2, Slb2Entry};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p raeen-firmware slb2`
Expected: PASS — `test result: ok. 3 passed`.

- [ ] **Step 5: Lint and commit**

```bash
cargo clippy -p raeen-firmware --all-targets
git add crates/raeen-firmware/src/slb2.rs crates/raeen-firmware/src/lib.rs
git commit -m "feat(firmware): parse SLB2 firmware container"
```

Expected clippy: zero warnings.

---

### Task 3: `Firmware` package API

**Files:**
- Create: `crates/raeen-firmware/src/pup.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod pup;` and re-export `Firmware`)

**Interfaces:**
- Consumes: `slb2::{parse_slb2, Slb2Entry}` from Task 2.
- Produces:
  - `pub struct Firmware` (holds mmap or in-memory bytes + parsed entries)
  - `pub fn Firmware::open(path: impl AsRef<std::path::Path>) -> Result<Firmware, FirmwareError>`
  - `pub fn Firmware::from_bytes(bytes: Vec<u8>) -> Result<Firmware, FirmwareError>`
  - `pub fn Firmware::entries(&self) -> &[Slb2Entry]`
  - `pub fn Firmware::read_entry(&self, index: usize) -> Result<&[u8], FirmwareError>`

- [ ] **Step 1: Write the failing tests**

Add `pub mod pup;` to `lib.rs`, then create `crates/raeen-firmware/src/pup.rs` with only this test module (implementation in Step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use raeen_core::error::FirmwareError;

    /// SLB2 with one entry whose 4-byte payload ("DATA") sits at block 2.
    fn synthetic_firmware() -> Vec<u8> {
        let payload_off = 2 * 0x200usize;
        let mut buf = vec![0u8; payload_off + 4];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[0x0C..0x10].copy_from_slice(&1u32.to_le_bytes()); // file_count = 1
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes()); // block_offset = 2
        buf[0x24..0x28].copy_from_slice(&4u32.to_le_bytes()); // size = 4
        buf[0x30..0x30 + 12].copy_from_slice(b"PS5UPDATE1.P"); // truncated name ok
        buf[payload_off..payload_off + 4].copy_from_slice(b"DATA");
        buf
    }

    #[test]
    fn from_bytes_enumerates_entries() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        assert_eq!(fw.entries().len(), 1);
        assert_eq!(fw.entries()[0].size, 4);
    }

    #[test]
    fn read_entry_returns_payload() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        assert_eq!(fw.read_entry(0).unwrap(), b"DATA");
    }

    #[test]
    fn read_entry_out_of_range_index_errors() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        assert!(matches!(
            fw.read_entry(5),
            Err(FirmwareError::PupEntryOutOfBounds { index: 5 })
        ));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p raeen-firmware pup`
Expected: FAIL to compile — `cannot find type Firmware`.

- [ ] **Step 3: Write the implementation**

At the top of `crates/raeen-firmware/src/pup.rs`:

```rust
//! Firmware package access — opens an SLB2/PUP container and exposes entries.
//!
//! The real `PS5UPDATE.PUP` is ~1.2 GB, so [`Firmware::open`] memory-maps the
//! file rather than reading it into RAM. Reads return borrowed slices into the
//! mapping; no entry payload is copied or decrypted here.

use crate::slb2::{parse_slb2, Slb2Entry};
use std::path::Path;
use raeen_core::error::{FirmwareError, LoaderError};

enum Backing {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Bytes(b) => b,
        }
    }
}

/// An opened PS5 firmware package (SLB2 container).
pub struct Firmware {
    backing: Backing,
    entries: Vec<Slb2Entry>,
}

impl Firmware {
    /// Open a firmware package from a file path (memory-mapped, read-only).
    ///
    /// # Errors
    ///
    /// I/O failures surface as [`FirmwareError::Loader`]; a bad container
    /// surfaces as [`FirmwareError::InvalidPupMagic`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FirmwareError> {
        let file = std::fs::File::open(path.as_ref()).map_err(LoaderError::from)?;
        // SAFETY: opened read-only; treated as an immutable byte slice for the
        // lifetime of the mapping. We never mutate through it.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(LoaderError::from)?;
        let entries = parse_slb2(&mmap)?;
        Ok(Self {
            backing: Backing::Mmap(mmap),
            entries,
        })
    }

    /// Construct from an in-memory buffer (tests, piped input).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, FirmwareError> {
        let entries = parse_slb2(&bytes)?;
        Ok(Self {
            backing: Backing::Bytes(bytes),
            entries,
        })
    }

    /// The container's entries.
    pub fn entries(&self) -> &[Slb2Entry] {
        &self.entries
    }

    /// Read the raw (possibly encrypted) bytes of entry `index`.
    ///
    /// # Errors
    ///
    /// [`FirmwareError::PupEntryOutOfBounds`] if `index` is invalid or the
    /// entry's declared range falls outside the container.
    pub fn read_entry(&self, index: usize) -> Result<&[u8], FirmwareError> {
        let entry = self
            .entries
            .get(index)
            .ok_or(FirmwareError::PupEntryOutOfBounds { index })?;
        let data = self.backing.as_slice();
        let start = entry.offset as usize;
        let end = start
            .checked_add(entry.size as usize)
            .filter(|&e| e <= data.len())
            .ok_or(FirmwareError::PupEntryOutOfBounds { index })?;
        Ok(&data[start..end])
    }
}
```

Then in `crates/raeen-firmware/src/lib.rs` add:

```rust
pub mod pup;

pub use pup::Firmware;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p raeen-firmware pup`
Expected: PASS — `test result: ok. 3 passed`.

- [ ] **Step 5: Lint and commit**

```bash
cargo clippy -p raeen-firmware --all-targets
git add crates/raeen-firmware/src/pup.rs crates/raeen-firmware/src/lib.rs
git commit -m "feat(firmware): add Firmware package open/read API"
```

Expected clippy: zero warnings.

---

### Task 4: `KeyProvider` decryption seam

**Files:**
- Create: `crates/raeen-firmware/src/crypto/mod.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod crypto;` and re-exports)

**Interfaces:**
- Produces:
  - `pub struct KeyRequest { pub key_type: u32, pub key_id: u64 }` (derives `Debug, Clone, Copy, PartialEq, Eq`)
  - `pub struct SegmentKey { pub key: [u8; 16], pub iv: [u8; 16] }` (derives `Debug, Clone, PartialEq, Eq`)
  - `pub trait KeyProvider: Send + Sync { fn segment_key(&self, req: &KeyRequest) -> Option<SegmentKey>; }`
  - `pub struct NoKeysProvider;` implementing `KeyProvider` (always `None`)
  - `pub fn require_key(provider: &dyn KeyProvider, req: &KeyRequest) -> Result<SegmentKey, FirmwareError>`

**Boundary reminder (Global Constraints):** this module *consumes* keys only. It contains no keys and no key derivation.

- [ ] **Step 1: Write the failing tests**

Add `pub mod crypto;` to `lib.rs`, then create `crates/raeen-firmware/src/crypto/mod.rs` with only this test module (implementation in Step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use raeen_core::error::FirmwareError;

    #[test]
    fn no_keys_provider_returns_none() {
        let req = KeyRequest { key_type: 1, key_id: 0xABCD };
        assert_eq!(NoKeysProvider.segment_key(&req), None);
    }

    #[test]
    fn require_key_maps_missing_to_error() {
        let req = KeyRequest { key_type: 1, key_id: 0xABCD };
        let err = require_key(&NoKeysProvider, &req).unwrap_err();
        assert!(matches!(err, FirmwareError::MissingKey { key_id: 0xABCD }));
    }

    #[test]
    fn require_key_returns_supplied_key() {
        struct FixedProvider;
        impl KeyProvider for FixedProvider {
            fn segment_key(&self, _req: &KeyRequest) -> Option<SegmentKey> {
                Some(SegmentKey { key: [1u8; 16], iv: [2u8; 16] })
            }
        }
        let req = KeyRequest { key_type: 0, key_id: 7 };
        let key = require_key(&FixedProvider, &req).unwrap();
        assert_eq!(key.key, [1u8; 16]);
        assert_eq!(key.iv, [2u8; 16]);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p raeen-firmware crypto`
Expected: FAIL to compile — `cannot find type KeyRequest`.

- [ ] **Step 3: Write the implementation**

At the top of `crates/raeen-firmware/src/crypto/mod.rs`:

```rust
//! The decryption boundary.
//!
//! Raeen ships **no keys** and no key-extraction tooling. All decryption is
//! driven by a user-supplied [`KeyProvider`]; the default [`NoKeysProvider`]
//! returns nothing and decryption fails cleanly with
//! [`FirmwareError::MissingKey`]. This module consumes keys — it never
//! derives, guesses, brute-forces, or extracts them.

use raeen_core::error::FirmwareError;

/// Identifies which key a SELF/module segment needs, read from its metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRequest {
    /// Key type from the SELF/segment header.
    pub key_type: u32,
    /// Key id / seed identifying the specific key.
    pub key_id: u64,
}

/// A content key + IV supplied by a user [`KeyProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentKey {
    /// 128-bit content key.
    pub key: [u8; 16],
    /// 128-bit initialization vector.
    pub iv: [u8; 16],
}

/// User-supplied source of decryption keys.
///
/// Implementors load keys the user obtained from hardware they own. The
/// default [`NoKeysProvider`] supplies none.
pub trait KeyProvider: Send + Sync {
    /// Return the key for `req`, or `None` if unavailable.
    fn segment_key(&self, req: &KeyRequest) -> Option<SegmentKey>;
}

/// Default provider that holds no keys. Decryption through it always fails
/// cleanly with [`FirmwareError::MissingKey`].
pub struct NoKeysProvider;

impl KeyProvider for NoKeysProvider {
    fn segment_key(&self, _req: &KeyRequest) -> Option<SegmentKey> {
        None
    }
}

/// Resolve a key or produce the canonical [`FirmwareError::MissingKey`].
///
/// Callers should treat the error as a normal, expected condition (log at
/// `info`, not `error`).
pub fn require_key(
    provider: &dyn KeyProvider,
    req: &KeyRequest,
) -> Result<SegmentKey, FirmwareError> {
    provider
        .segment_key(req)
        .ok_or(FirmwareError::MissingKey { key_id: req.key_id })
}
```

Then in `crates/raeen-firmware/src/lib.rs` add:

```rust
pub mod crypto;

pub use crypto::{require_key, KeyProvider, KeyRequest, NoKeysProvider, SegmentKey};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p raeen-firmware crypto`
Expected: PASS — `test result: ok. 3 passed`.

- [ ] **Step 5: Lint and commit**

```bash
cargo clippy -p raeen-firmware --all-targets
git add crates/raeen-firmware/src/crypto/mod.rs crates/raeen-firmware/src/lib.rs
git commit -m "feat(firmware): add user-supplied KeyProvider decryption seam"
```

Expected clippy: zero warnings.

---

### Task 5: SCE NID hashing and encoding

**Files:**
- Create: `crates/raeen-firmware/src/dynlib/mod.rs`
- Create: `crates/raeen-firmware/src/dynlib/nid.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod dynlib;`)

**Interfaces:**
- Produces (in `dynlib::nid`):
  - `pub fn nid_of(name: &str) -> u64`
  - `pub fn encode_nid(nid: u64) -> String`
  - `pub fn decode_nid(s: &str) -> Option<u64>`

**Background:** Sony replaces symbol names with NIDs. `nid = SHA-1(name ‖ SCE_NID_SALT)[0..8]` interpreted little-endian; the string form base64-encodes those 8 bytes with a custom alphabet. The salt is a public, community-documented constant — a salt, not a secret key.

- [ ] **Step 1: Write the failing tests**

Add `pub mod dynlib;` to `lib.rs`. Create `crates/raeen-firmware/src/dynlib/mod.rs`:

```rust
//! Sony dynamic-linking data: NID hashing now; import/export/relocation
//! parsing and the NID linker arrive in the LM1 plan.

pub mod nid;
```

Create `crates/raeen-firmware/src/dynlib/nid.rs` with only this test module (implementation in Step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nid_is_deterministic() {
        assert_eq!(nid_of("sceKernelError"), nid_of("sceKernelError"));
    }

    #[test]
    fn distinct_names_give_distinct_nids() {
        assert_ne!(nid_of("foo"), nid_of("bar"));
    }

    #[test]
    fn encode_decode_round_trips() {
        for nid in [0u64, 1, 0x0123_4567_89AB_CDEF, u64::MAX] {
            let s = encode_nid(nid);
            assert_eq!(s.len(), 11, "NID string is 11 chars");
            assert_eq!(decode_nid(&s), Some(nid), "round-trip {nid:#x}");
        }
    }

    #[test]
    fn alphabet_endpoints() {
        // 6-bit index 0 -> 'A', 63 -> '-'. Verify via single-value encodings:
        // nid whose top 6 bits are 0 begins with 'A'.
        assert!(encode_nid(0).starts_with('A'));
        // decode rejects characters outside the alphabet.
        assert_eq!(decode_nid("!!!!!!!!!!!"), None);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p raeen-firmware nid`
Expected: FAIL to compile — `cannot find function nid_of`.

- [ ] **Step 3: Write the implementation**

At the top of `crates/raeen-firmware/src/dynlib/nid.rs`:

```rust
//! SCE NID (Name ID) hashing and base64 encoding.

use sha1::{Digest, Sha1};

/// Public, community-documented 16-byte suffix appended to a symbol name
/// before hashing. This is a salt, not a secret key.
const SCE_NID_SALT: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Custom base64 alphabet used for NID strings.
const NID_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

/// Compute the 64-bit NID for a symbol name.
pub fn nid_of(name: &str) -> u64 {
    let mut hasher = Sha1::new();
    hasher.update(name.as_bytes());
    hasher.update(SCE_NID_SALT);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[0..8].try_into().unwrap())
}

/// Encode a 64-bit NID into its 11-character SCE base64 string.
///
/// Bytes are taken little-endian and packed MSB-first, 6 bits per character,
/// no padding (64 bits → 11 characters, the last carrying 4 significant bits).
pub fn encode_nid(nid: u64) -> String {
    let bytes = nid.to_le_bytes();
    let mut out = String::with_capacity(11);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &b in &bytes {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 6 {
            bits -= 6;
            out.push(NID_ALPHABET[((acc >> bits) & 0x3F) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(NID_ALPHABET[((acc << (6 - bits)) & 0x3F) as usize] as char);
    }
    out
}

/// Decode an SCE base64 NID string back into a 64-bit NID.
///
/// Returns `None` if the string contains a character outside the alphabet or
/// decodes to fewer than 8 bytes.
pub fn decode_nid(s: &str) -> Option<u64> {
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    let mut bytes = Vec::with_capacity(8);
    for ch in s.chars() {
        let val = NID_ALPHABET.iter().position(|&a| a as char == ch)? as u64;
        acc = (acc << 6) | val;
        bits += 6;
        while bits >= 8 {
            bits -= 8;
            bytes.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    if bytes.len() < 8 {
        return None;
    }
    Some(u64::from_le_bytes(bytes[0..8].try_into().ok()?))
}
```

Then add `pub mod dynlib;` to `crates/raeen-firmware/src/lib.rs` (if not already present from Step 1).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p raeen-firmware nid`
Expected: PASS — `test result: ok. 4 passed`.

- [ ] **Step 5: Verify against a public NID vector (correctness gate)**

The round-trip tests prove internal consistency but not that our bit order matches Sony's. Cross-check one real symbol against a public NID database (e.g. a community LibKernel `.nid`/`.rdb` list): compute `encode_nid(nid_of("<known symbol>"))` and confirm it equals the published NID string for that symbol. If it differs, the discrepancy is the base64 bit order or the byte endianness in `nid_of` — adjust and re-run. Record the confirmed `(symbol, nid_string)` pair as a pinned test:

```rust
    #[test]
    fn matches_public_nid_vector() {
        // Replace with the verified pair from a public NID database.
        assert_eq!(encode_nid(nid_of("<verified symbol>")), "<verified nid>");
    }
```

- [ ] **Step 6: Lint and commit**

```bash
cargo clippy -p raeen-firmware --all-targets
git add crates/raeen-firmware/src/dynlib/mod.rs crates/raeen-firmware/src/dynlib/nid.rs crates/raeen-firmware/src/lib.rs
git commit -m "feat(firmware): add SCE NID hashing and base64 encoding"
```

Expected clippy: zero warnings.

---

### Task 6: `--firmware-info` diagnostic (LM0 acceptance)

**Files:**
- Create: `crates/raeen-firmware/src/report.rs`
- Modify: `crates/raeen-firmware/src/lib.rs` (add `pub mod report;` and re-export)
- Modify: `crates/raeen-gui/Cargo.toml` (add `raeen-firmware` dependency)
- Modify: `crates/raeen-gui/src/main.rs` (handle `--firmware-info <path>` before launching the GUI)

**Interfaces:**
- Consumes: `Firmware` from Task 3.
- Produces: `pub fn summarize(firmware: &Firmware) -> String`.

- [ ] **Step 1: Write the failing test**

Add `pub mod report;` to `crates/raeen-firmware/src/lib.rs`, then create `crates/raeen-firmware/src/report.rs` with only this test module (implementation in Step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Firmware;

    fn synthetic_firmware() -> Vec<u8> {
        let mut buf = vec![0u8; 0x20 + 0x30];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[0x0C..0x10].copy_from_slice(&1u32.to_le_bytes());
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes());
        buf[0x24..0x28].copy_from_slice(&0x100u32.to_le_bytes());
        buf[0x30..0x30 + 14].copy_from_slice(b"PS5UPDATE1.PUP");
        buf
    }

    #[test]
    fn summary_lists_entries() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        let text = summarize(&fw);
        assert!(text.contains("1 entry"));
        assert!(text.contains("PS5UPDATE1.PUP"));
        assert!(text.contains("0x400")); // offset 2 * 0x200
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p raeen-firmware report`
Expected: FAIL to compile — `cannot find function summarize`.

- [ ] **Step 3: Write the implementation**

At the top of `crates/raeen-firmware/src/report.rs`:

```rust
//! Human-readable firmware inspection report (drives `--firmware-info`).

use crate::Firmware;
use std::fmt::Write;

/// Render a plaintext summary of a firmware container's entries.
pub fn summarize(firmware: &Firmware) -> String {
    let entries = firmware.entries();
    let mut s = String::new();
    let plural = if entries.len() == 1 { "entry" } else { "entries" };
    let _ = writeln!(s, "SLB2 firmware container: {} {}", entries.len(), plural);
    for (i, e) in entries.iter().enumerate() {
        let _ = writeln!(
            s,
            "  [{i}] {:<20} offset={:#x} size={} bytes (encrypted payload; not decrypted)",
            e.name, e.offset, e.size
        );
    }
    s
}
```

Then add to `crates/raeen-firmware/src/lib.rs`:

```rust
pub mod report;

pub use report::summarize;
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p raeen-firmware report`
Expected: PASS — `test result: ok. 1 passed`.

- [ ] **Step 5: Add the firmware dependency to the GUI crate**

In `crates/raeen-gui/Cargo.toml`, add under `[dependencies]` (alongside the other `raeen-*` entries):

```toml
raeen-firmware = { workspace = true }
```

- [ ] **Step 6: Wire the CLI flag into `main`**

In `crates/raeen-gui/src/main.rs`, insert this block immediately after `raeen_core::logging::init("info");` and before the banner `info!` lines:

```rust
    // Diagnostic: `raeen --firmware-info <PUP>` inspects a firmware package
    // and exits without launching the GUI. It never decrypts anything.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--firmware-info") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--firmware-info requires a path to a PUP file"))?;
        let firmware = raeen_firmware::Firmware::open(path)?;
        print!("{}", raeen_firmware::summarize(&firmware));
        return Ok(());
    }
```

- [ ] **Step 7: Build the workspace and run the diagnostic test suite**

Run: `cargo build -p raeen-gui`
Expected: compiles cleanly.

Run: `cargo test -p raeen-firmware`
Expected: all tests pass (`slb2`, `pup`, `crypto`, `nid`, `report`, plus the Task 1 smoke test).

- [ ] **Step 8: Manual LM0 acceptance against the real firmware**

Run: `cargo run -p raeen-gui -- --firmware-info "PS5 Firmware/PS5UPDATE.PUP"`
Expected: prints the SLB2 container summary — at least one entry whose name contains `PS5UPDATE`, with a non-zero offset and size — then exits without launching the GUI and without attempting decryption. If entry names look wrong (garbled/empty), the real SLB2 field offsets differ from the documented layout; adjust `slb2.rs` field positions against the observed bytes and re-run (the synthetic unit tests still guard the parser logic).

- [ ] **Step 9: Lint and commit**

```bash
cargo clippy -p raeen-firmware -p raeen-gui --all-targets
git add crates/raeen-firmware/src/report.rs crates/raeen-firmware/src/lib.rs crates/raeen-gui/Cargo.toml crates/raeen-gui/src/main.rs Cargo.lock
git commit -m "feat(firmware): add --firmware-info diagnostic (LM0)"
```

Expected clippy: zero warnings.

---

## Self-Review

**Spec coverage (against `2026-07-12-raeen-lle-firmware-spine-design.md`):**
- §3.1 new `raeen-firmware` crate → Task 1. ✓
- §3.3.1 `pup.rs` PUP parser + `Firmware::open/entries/read_entry` → Tasks 2–3. (Spec's `read_entry(&PupEntry)` is realized as `read_entry(index)` — cleaner error reporting; documented in interfaces.) ✓
- §3.3.2 `crypto/` `KeyProvider` + `NoKeysProvider` + missing-key error → Task 4. ✓
- §3.3.4 `dynlib/nid.rs` NID hashing + name↔NID basis → Task 5. ✓
- §5 step 3 `--firmware-info` CLI (LM0 acceptance) → Task 6. ✓
- **Deferred to the LM1 plan (out of scope here, per spec §7 "parse-what's-readable now"):** `crypto/self_crypto.rs` SELF segment decryption machinery (§3.3.2), `sprx.rs` module parser (§3.3.3), `dynlib/mod.rs` `PT_SCE_DYNLIBDATA` parse + `dynlib/linker.rs` (§3.3.4), `registry.rs` HLE/LLE dispatch (§3.3.5), and the LM1 end-to-end module-load test. These need real/homebrew module bytes to verify formats and cannot be written as complete, correct code today. This plan delivers their foundation (crate, PUP access, key seam, NID library) so LM1 is unblocked.

**Placeholder scan:** No `TBD`/`TODO`/"handle edge cases". The one deliberately-unpinned value is the public NID vector in Task 5 Step 5 — it is a *verification step with explicit instructions*, not a code placeholder, because the correct vector must come from a public database rather than be invented here.

**Type consistency:** `Firmware`, `Slb2Entry`, `KeyRequest`, `SegmentKey`, `KeyProvider`, `NoKeysProvider`, `require_key`, `nid_of`, `encode_nid`, `decode_nid`, `summarize` are named identically across their defining task and every consuming task. `read_entry(index: usize)` is index-based in both its definition (Task 3) and the `FirmwareError::PupEntryOutOfBounds { index }` it returns. Error variants match `raeen_core::error::FirmwareError` exactly.

---

## Follow-on: LM1 plan (not included here)

Once LM0 lands, the next plan covers LM1: `self_crypto` SELF decryption driven by the `KeyProvider`, `sprx.rs` (`ET_SCE_DYNLIB`), `dynlib` `PT_SCE_DYNLIBDATA` parsing (imports/exports/relocs), the NID linker, the HLE/LLE `ModuleRegistry` (bridged to `raeen-hle::HleRegistry` via a resolver trait to keep `raeen-firmware` free of an upward dependency), and an end-to-end test loading a homebrew/decrypted `.sprx` with imports resolved to HLE stubs.
