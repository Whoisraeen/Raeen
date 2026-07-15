//! SELF decrypt-or-passthrough.
//!
//! Recovers the inner ELF from a SELF (Signed ELF) container. Each segment
//! entry is either already plaintext — the homebrew / already-decrypted
//! path this milestone targets — and is copied through unchanged, or it is
//! flagged encrypted and is decrypted with a key obtained from a
//! user-supplied [`KeyProvider`].
//!
//! This module performs standard AES-128 given a key the caller supplies.
//! It never contains, hardcodes, derives, guesses, or brute-forces a key —
//! see the clean-room boundary in the design doc, §2. The only source of a
//! key is the `KeyProvider` passed into [`decrypt_self`].
//!
//! Layout assumption: each [`SelfEntry`] names a contiguous byte range
//! within the SELF file (`offset`..`offset + compressed_size`) holding that
//! segment's data (ciphertext if encrypted, plaintext otherwise); the
//! reassembled inner ELF is the concatenation, in entry order, of each
//! segment's plaintext bytes truncated to `uncompressed_size`. Segment
//! compression is out of scope for this milestone.

use crate::crypto::{KeyProvider, KeyRequest, SegmentKey, require_key};
use aes::Aes128;
use aes::cipher::{Array, BlockCipherEncrypt, BlockModeDecrypt, KeyInit, KeyIvInit};
use tracing::{debug, info};
use xps5x_core::error::{FirmwareError, LoaderError};
use xps5x_loader::self_format::{SelfEntry, SelfHeader};

/// SELF magic used by this crate's in-tree fixtures. The real-hardware magics
/// are accepted too — every container variant is recognized via
/// [`xps5x_loader::self_format::is_self_magic`], which owns the canonical list.
#[cfg(test)]
use xps5x_loader::self_format::SELF_MAGIC;

/// Fixture-only marker selecting AES-128-CBC for an encrypted segment.
///
/// Real PS4/PS5 SELF `properties` has no cipher selector — bits 1-3 are the
/// independent `encrypted`/`signed`/`compressed` flags, and real encrypted
/// segments are CTR. This project's CBC path is exercised only by in-tree
/// fixtures, so it is selected by a **reserved** bit that cannot collide with
/// any real flag (bit 12), rather than by re-reading the flag bits as an enum
/// (which is what made real `signed` plaintext segments look CBC-encrypted).
const PROP_FIXTURE_CBC: u64 = 1 << 12;

/// Fixed size of the SELF header, in bytes.
const SELF_HEADER_SIZE: usize = 32;

/// Fixed size of one SELF entry record, in bytes.
const SELF_ENTRY_SIZE: usize = 32;

/// AES block size, in bytes.
const BLOCK_SIZE: usize = 16;

/// The plaintext inner ELF recovered from a SELF, plus a tally of how many
/// segments were decrypted vs. passed through unchanged.
#[derive(Debug, Clone)]
pub struct DecryptedSelf {
    /// The inner ELF, ready for `sprx::parse_sprx`.
    pub elf: Vec<u8>,
    /// Number of segments that were AES-decrypted via the `KeyProvider`.
    pub decrypted_segments: usize,
    /// Number of segments that were already plaintext and copied through.
    pub passthrough_segments: usize,
}

/// Which AES mode a segment's properties field selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentCipher {
    Ctr,
    Cbc,
}

impl SegmentCipher {
    /// Decode from a [`SelfEntry::properties`] bit field. Only consulted for
    /// segments [`SelfEntry::is_encrypted`] already flagged (real bit 1).
    ///
    /// Real encrypted SELF segments are AES-128-CTR; the CBC path exists for
    /// in-tree fixtures and is selected by the reserved [`PROP_FIXTURE_CBC`]
    /// bit. This deliberately does *not* re-read bits 1-3 as an enum — those
    /// are the real format's independent encrypted/signed/compressed flags.
    fn from_properties(properties: u64) -> Self {
        if properties & PROP_FIXTURE_CBC != 0 {
            SegmentCipher::Cbc
        } else {
            SegmentCipher::Ctr
        }
    }
}

/// Recover the inner ELF from a SELF image.
///
/// Parses the SELF header and entry table, bounds-checked against `data` —
/// this never panics or indexes out of range on truncated/malformed input.
/// Each plaintext segment is copied through unchanged; each segment flagged
/// encrypted is routed through `provider` via [`require_key`] and decrypted
/// with AES-128 (CTR or CBC, selected per segment).
///
/// # Errors
///
/// - [`FirmwareError::MissingKey`] if an encrypted segment needs a key
///   `provider` does not have. This is a normal, expected, non-fatal
///   condition — callers should log it at `info`, not `error`.
/// - [`FirmwareError::MalformedSelf`] if the header, entry table, or a
///   segment's data range is truncated or otherwise inconsistent.
/// - [`FirmwareError::Loader`] (`InvalidSelfMagic`) if the magic doesn't
///   match.
pub fn decrypt_self(
    data: &[u8],
    provider: &dyn KeyProvider,
) -> Result<DecryptedSelf, FirmwareError> {
    let header = parse_header(data)?;
    let entries = parse_entries(data, &header)?;

    info!(
        "Decrypting SELF: {} entries, header_size={:#x}",
        entries.len(),
        header.header_size
    );

    // A real title's SELF stores each segment's data at the *entry's* offset,
    // and the inner ELF's own header+phdrs inline right after the entry table
    // — the phdr `p_offset`s describe the ORIGINAL (pre-SELF) layout, not where
    // the bytes sit in this file. Reassembling by concatenating entries (the
    // legacy model below) therefore produces a corrupt ELF for real titles.
    //
    // Blocked entries (properties bit 11) are the segment data, and their
    // `segment_index()` is the ELF phdr index they belong to; the non-blocked
    // entries are block/digest tables. If any entry is blocked this is a real
    // SELF, so reassemble properly.
    if entries.iter().any(|e| e.is_blocked()) {
        return reassemble_real_self(data, &header, &entries, provider);
    }

    let mut elf = Vec::new();
    let mut decrypted_segments = 0usize;
    let mut passthrough_segments = 0usize;

    for (index, entry) in entries.iter().enumerate() {
        if !entry.has_data() {
            debug!("SELF entry {index} has no data, skipping");
            continue;
        }

        let start = entry.offset as usize;
        let seg_len = entry.compressed_size as usize;
        let end = start.checked_add(seg_len).ok_or_else(|| {
            FirmwareError::MalformedSelf(format!("SELF entry {index} offset/size overflow"))
        })?;
        if end > data.len() {
            return Err(FirmwareError::MalformedSelf(format!(
                "SELF entry {index} data range [{start:#x}, {end:#x}) exceeds file size {:#x}",
                data.len()
            )));
        }
        let segment_data = &data[start..end];

        if entry.is_encrypted() {
            let req = KeyRequest {
                key_type: header.key_type,
                key_id: entry.segment_index(),
            };
            let key = require_key(provider, &req)?;
            let cipher = SegmentCipher::from_properties(entry.properties);
            let plaintext = decrypt_segment(segment_data, &key, cipher, index)?;
            let take = (entry.uncompressed_size as usize).min(plaintext.len());
            elf.extend_from_slice(&plaintext[..take]);
            decrypted_segments += 1;
            debug!("SELF entry {index}: decrypted {take} bytes ({cipher:?})");
        } else {
            let take = (entry.uncompressed_size as usize).min(segment_data.len());
            elf.extend_from_slice(&segment_data[..take]);
            passthrough_segments += 1;
            debug!("SELF entry {index}: passthrough {take} bytes");
        }
    }

    if elf.is_empty() {
        return Err(FirmwareError::MalformedSelf(
            "SELF contains no segment data to reassemble an inner ELF".to_string(),
        ));
    }

    info!(
        "SELF decrypted: {decrypted_segments} segment(s) decrypted, {passthrough_segments} passed through, {} ELF byte(s)",
        elf.len()
    );

    Ok(DecryptedSelf {
        elf,
        decrypted_segments,
        passthrough_segments,
    })
}

/// Reassemble the inner ELF of a **real** (hardware) SELF.
///
/// Layout, verified against a real PS5 title: a 0x20-byte header, then
/// `num_entries` 0x20-byte entries, then the inner ELF's header + program
/// headers inline. Each *blocked* entry (properties bit 11) carries one
/// segment's bytes at the entry's own `offset`, and its
/// [`SelfEntry::segment_index`] names the ELF phdr those bytes belong to. The
/// phdr's `p_offset` is where they go in the **reconstructed** ELF (the
/// original pre-SELF layout) — which is why concatenating entries, as the
/// legacy fixture path does, cannot work here.
///
/// Non-blocked entries are block/digest tables and carry no ELF bytes; they are
/// skipped. Segments that overlap a `PT_LOAD` (e.g. `PT_DYNAMIC`,
/// `GNU_EH_FRAME`) have no entry of their own — their bytes arrive with the
/// containing `PT_LOAD`, which is why the output is addressed by `p_offset`
/// rather than per-segment.
fn reassemble_real_self(
    data: &[u8],
    header: &SelfHeader,
    entries: &[SelfEntry],
    provider: &dyn KeyProvider,
) -> Result<DecryptedSelf, FirmwareError> {
    let elf_start = SELF_HEADER_SIZE + entries.len() * SELF_ENTRY_SIZE;
    let ehdr = data.get(elf_start..).ok_or_else(|| {
        FirmwareError::MalformedSelf(format!("SELF entry table overruns file ({elf_start:#x})"))
    })?;
    let phdrs = parse_inner_phdrs(ehdr)?;

    // Size the image to the furthest byte any phdr claims, so every segment's
    // `p_offset` lands inside it. Any hole (a phdr with no entry that is not
    // covered by a PT_LOAD) stays zero rather than shifting later bytes.
    let mut elf_len = ehdr_and_phdr_span(ehdr)?;
    for p in &phdrs {
        let end = p.offset.saturating_add(p.filesz);
        elf_len = elf_len.max(usize::try_from(end).unwrap_or(usize::MAX));
    }
    let mut elf = vec![0u8; elf_len];

    // The ELF header + phdr table sit inline in the SELF; copy them verbatim.
    let head_len = ehdr_and_phdr_span(ehdr)?;
    elf[..head_len].copy_from_slice(&ehdr[..head_len]);

    // A SELF does not carry the inner ELF's section table — only its segments.
    // Real titles are left with an `e_shoff` describing the original,
    // pre-SELF file (one real eboot's pointed ~2.5 GB into a 254 MB image), so
    // the reconstructed ELF would advertise a section table that isn't there
    // and any strict whole-file parse rejects it. Sections are irrelevant to
    // loading (only program headers map anything), so emit an honest "no
    // sections" header instead of a dangling pointer.
    let e_shoff = u64::from_le_bytes(elf[0x28..0x30].try_into().unwrap());
    let e_shnum = u16::from_le_bytes(elf[0x3C..0x3E].try_into().unwrap());
    let sh_end = e_shoff.saturating_add(
        u64::from(u16::from_le_bytes(elf[0x3A..0x3C].try_into().unwrap()))
            .saturating_mul(u64::from(e_shnum)),
    );
    if e_shoff != 0 && sh_end > elf_len as u64 {
        debug!(
            "SELF inner ELF section table ({e_shnum} @ {e_shoff:#x}) is outside the reassembled \
             image ({elf_len:#x}) — the SELF did not preserve it; clearing e_shoff/e_shnum"
        );
        elf[0x28..0x30].copy_from_slice(&0u64.to_le_bytes()); // e_shoff
        elf[0x3C..0x3E].copy_from_slice(&0u16.to_le_bytes()); // e_shnum
        elf[0x3E..0x40].copy_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    }

    let mut decrypted_segments = 0usize;
    let mut passthrough_segments = 0usize;

    for (index, entry) in entries.iter().enumerate() {
        if !entry.is_blocked() || !entry.has_data() {
            debug!(
                "SELF entry {index}: props={:#x} not a blocked segment (block/digest table) — skipped",
                entry.properties
            );
            continue;
        }
        if entry.is_compressed() {
            return Err(FirmwareError::MalformedSelf(format!(
                "SELF entry {index} is compressed; segment decompression is not implemented"
            )));
        }

        let idx = usize::try_from(entry.segment_index()).unwrap_or(usize::MAX);
        let Some(phdr) = phdrs.get(idx) else {
            return Err(FirmwareError::MalformedSelf(format!(
                "SELF entry {index} names phdr {idx}, but the inner ELF has {} program header(s)",
                phdrs.len()
            )));
        };

        let start = usize::try_from(entry.offset).unwrap_or(usize::MAX);
        let len = usize::try_from(entry.compressed_size).unwrap_or(usize::MAX);
        let end = start.checked_add(len).ok_or_else(|| {
            FirmwareError::MalformedSelf(format!("SELF entry {index} offset/size overflow"))
        })?;
        let src = data.get(start..end).ok_or_else(|| {
            FirmwareError::MalformedSelf(format!(
                "SELF entry {index} data [{start:#x}, {end:#x}) exceeds file size {:#x}",
                data.len()
            ))
        })?;

        let bytes = if entry.is_encrypted() {
            let req = KeyRequest {
                key_type: header.key_type,
                key_id: entry.segment_index(),
            };
            let key = require_key(provider, &req)?;
            let cipher = SegmentCipher::from_properties(entry.properties);
            decrypted_segments += 1;
            decrypt_segment(src, &key, cipher, index)?
        } else {
            passthrough_segments += 1;
            src.to_vec()
        };

        let dst_start = usize::try_from(phdr.offset).unwrap_or(usize::MAX);
        let take = usize::try_from(phdr.filesz)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let dst = elf
            .get_mut(dst_start..dst_start.saturating_add(take))
            .ok_or_else(|| {
                FirmwareError::MalformedSelf(format!(
                    "SELF entry {index} -> phdr {idx} p_offset {dst_start:#x}+{take:#x} exceeds the reassembled image ({elf_len:#x})"
                ))
            })?;
        dst.copy_from_slice(&bytes[..take]);
        debug!(
            "SELF entry {index}: phdr {idx} <- {take:#x} byte(s) from {start:#x} to p_offset {dst_start:#x}"
        );
    }

    info!(
        "SELF reassembled (real layout): {decrypted_segments} decrypted, {passthrough_segments} passed through, {} phdr(s), {} ELF byte(s)",
        phdrs.len(),
        elf.len()
    );

    Ok(DecryptedSelf {
        elf,
        decrypted_segments,
        passthrough_segments,
    })
}

/// One inner-ELF program header, as far as reassembly cares.
struct InnerPhdr {
    offset: u64,
    filesz: u64,
}

/// Byte span of the inner ELF's header plus its program-header table.
fn ehdr_and_phdr_span(ehdr: &[u8]) -> Result<usize, FirmwareError> {
    let (e_phoff, e_phentsize, e_phnum) = inner_phdr_table(ehdr)?;
    let end = e_phoff
        .saturating_add((e_phentsize as u64).saturating_mul(e_phnum as u64))
        .max(0x40);
    usize::try_from(end)
        .map_err(|_| FirmwareError::MalformedSelf("inner ELF phdr table overflows".to_string()))
}

/// Read `(e_phoff, e_phentsize, e_phnum)` from the inner ELF header.
///
/// Parsed by hand rather than via a full ELF parse: a SELF's inner ELF
/// routinely has a stripped/out-of-file `e_shoff` (a real title's pointed ~2.5
/// GB into a 254 MB file), which makes a strict whole-file parse fail even
/// though every program header is perfectly valid. Reassembly only needs the
/// phdr table.
fn inner_phdr_table(ehdr: &[u8]) -> Result<(u64, u16, u16), FirmwareError> {
    if ehdr.len() < 0x40 || ehdr[0..4] != [0x7F, b'E', b'L', b'F'] {
        return Err(FirmwareError::MalformedSelf(
            "SELF inner ELF header missing or not an ELF".to_string(),
        ));
    }
    let e_phoff = u64::from_le_bytes(ehdr[0x20..0x28].try_into().unwrap());
    let e_phentsize = u16::from_le_bytes(ehdr[0x36..0x38].try_into().unwrap());
    let e_phnum = u16::from_le_bytes(ehdr[0x38..0x3A].try_into().unwrap());
    if e_phentsize < 56 {
        return Err(FirmwareError::MalformedSelf(format!(
            "SELF inner ELF e_phentsize {e_phentsize} is too small for a 64-bit phdr"
        )));
    }
    Ok((e_phoff, e_phentsize, e_phnum))
}

/// Parse the inner ELF's program headers (offset/filesz only).
fn parse_inner_phdrs(ehdr: &[u8]) -> Result<Vec<InnerPhdr>, FirmwareError> {
    let (e_phoff, e_phentsize, e_phnum) = inner_phdr_table(ehdr)?;
    let mut out = Vec::with_capacity(e_phnum as usize);
    for i in 0..e_phnum as u64 {
        let at = usize::try_from(e_phoff + i * e_phentsize as u64).unwrap_or(usize::MAX);
        let p = ehdr.get(at..at.saturating_add(56)).ok_or_else(|| {
            FirmwareError::MalformedSelf(format!(
                "SELF inner ELF program header {i} at {at:#x} is outside the file"
            ))
        })?;
        out.push(InnerPhdr {
            offset: u64::from_le_bytes(p[0x08..0x10].try_into().unwrap()),
            filesz: u64::from_le_bytes(p[0x20..0x28].try_into().unwrap()),
        });
    }
    Ok(out)
}

/// Parse and bounds-check the fixed 32-byte SELF header.
fn parse_header(data: &[u8]) -> Result<SelfHeader, FirmwareError> {
    if data.len() < SELF_HEADER_SIZE {
        return Err(FirmwareError::MalformedSelf(format!(
            "SELF header truncated: {} byte(s), need at least {SELF_HEADER_SIZE}",
            data.len()
        )));
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if !xps5x_loader::self_format::is_self_magic(magic) {
        return Err(LoaderError::InvalidSelfMagic(magic).into());
    }

    let version = data[4];
    let mode = data[5];
    let endian = data[6];
    let attributes = data[7];
    let key_type = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let header_size = u16::from_le_bytes(data[12..14].try_into().unwrap());
    let meta_size = u16::from_le_bytes(data[14..16].try_into().unwrap());
    let file_size = u64::from_le_bytes(data[16..24].try_into().unwrap());
    let num_entries = u16::from_le_bytes(data[24..26].try_into().unwrap());
    let flags = u16::from_le_bytes(data[26..28].try_into().unwrap());

    Ok(SelfHeader {
        magic,
        version,
        mode,
        endian,
        attributes,
        key_type,
        header_size,
        meta_size,
        file_size,
        num_entries,
        flags,
        _padding: 0,
    })
}

/// Parse and bounds-check the SELF entry table that follows the header.
fn parse_entries(data: &[u8], header: &SelfHeader) -> Result<Vec<SelfEntry>, FirmwareError> {
    let num_entries = header.num_entries as usize;

    // Guard against an attacker-controlled `num_entries` driving a huge
    // pre-allocation before we've verified the entry table actually fits —
    // the LM0 SLB2 `file_count` fix is the standard to follow here.
    let available = data.len().saturating_sub(SELF_HEADER_SIZE);
    let max_entries = available / SELF_ENTRY_SIZE;
    if num_entries > max_entries {
        return Err(FirmwareError::MalformedSelf(format!(
            "SELF declares {num_entries} entries but only {max_entries} fit in {} byte(s)",
            data.len()
        )));
    }

    let mut entries = Vec::with_capacity(num_entries);
    for i in 0..num_entries {
        let base = SELF_HEADER_SIZE + i * SELF_ENTRY_SIZE;
        if base + SELF_ENTRY_SIZE > data.len() {
            return Err(FirmwareError::MalformedSelf(format!(
                "SELF entry {i} extends beyond file bounds"
            )));
        }
        let properties = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let offset = u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
        let compressed_size = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
        let uncompressed_size = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());
        entries.push(SelfEntry {
            properties,
            offset,
            compressed_size,
            uncompressed_size,
        });
    }
    Ok(entries)
}

/// Decrypt one segment's ciphertext with the supplied key.
fn decrypt_segment(
    ciphertext: &[u8],
    key: &SegmentKey,
    cipher: SegmentCipher,
    index: usize,
) -> Result<Vec<u8>, FirmwareError> {
    if !ciphertext.len().is_multiple_of(BLOCK_SIZE) {
        return Err(FirmwareError::MalformedSelf(format!(
            "SELF entry {index} ciphertext length {} is not a multiple of the AES block size ({BLOCK_SIZE})",
            ciphertext.len()
        )));
    }
    match cipher {
        SegmentCipher::Ctr => Ok(aes128_ctr_xor(key.key, key.iv, ciphertext)),
        SegmentCipher::Cbc => Ok(aes128_cbc_decrypt(key.key, key.iv, ciphertext)),
    }
}

/// AES-128-CTR keystream XOR. Encryption and decryption are the same
/// operation: encrypt the big-endian counter, XOR with the data, increment.
fn aes128_ctr_xor(key: [u8; 16], iv: [u8; 16], data: &[u8]) -> Vec<u8> {
    let cipher = Aes128::new(&key.into());
    let mut counter = iv;
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks(BLOCK_SIZE) {
        let mut keystream: Array<u8, _> = counter.into();
        cipher.encrypt_block(&mut keystream);
        for (b, k) in chunk.iter().zip(keystream.iter()) {
            out.push(b ^ k);
        }
        increment_counter(&mut counter);
    }
    out
}

/// Increment a 128-bit big-endian counter in place, with carry.
fn increment_counter(counter: &mut [u8; 16]) {
    for byte in counter.iter_mut().rev() {
        let (next, carry) = byte.overflowing_add(1);
        *byte = next;
        if !carry {
            break;
        }
    }
}

/// AES-128-CBC decrypt, block by block. SELF segments are framed to whole
/// AES blocks, so no padding scheme is applied or required.
fn aes128_cbc_decrypt(key: [u8; 16], iv: [u8; 16], data: &[u8]) -> Vec<u8> {
    let mut decryptor = cbc::Decryptor::<Aes128>::new(&key.into(), &iv.into());
    let mut out = data.to_vec();
    for chunk in out.chunks_mut(BLOCK_SIZE) {
        // `chunk` is exactly `BLOCK_SIZE` bytes: `data.len()` was already
        // checked to be a multiple of `BLOCK_SIZE` by `decrypt_segment`.
        let block: &mut Array<u8, _> = chunk.try_into().expect("chunk is one AES block");
        decryptor.decrypt_block(block);
    }
    out
}

/// AES-128-CBC encrypt, block by block. Only used by tests to build
/// synthetic encrypted SELF fixtures — production code never encrypts.
#[cfg(test)]
fn aes128_cbc_encrypt(key: [u8; 16], iv: [u8; 16], data: &[u8]) -> Vec<u8> {
    use aes::cipher::BlockModeEncrypt;
    let mut encryptor = cbc::Encryptor::<Aes128>::new(&key.into(), &iv.into());
    let mut out = data.to_vec();
    for chunk in out.chunks_mut(BLOCK_SIZE) {
        let block: &mut Array<u8, _> = chunk.try_into().expect("chunk is one AES block");
        encryptor.encrypt_block(block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::NoKeysProvider;

    const TEST_KEY: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F,
    ];
    const TEST_IV: [u8; 16] = [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
        0x1F,
    ];
    const TEST_KEY_ID: u64 = 7;

    /// A stub provider that returns a single known test key for
    /// `TEST_KEY_ID`, and nothing for anything else.
    struct StubProvider;
    impl KeyProvider for StubProvider {
        fn segment_key(&self, req: &KeyRequest) -> Option<SegmentKey> {
            if req.key_id == TEST_KEY_ID {
                Some(SegmentKey {
                    key: TEST_KEY,
                    iv: TEST_IV,
                })
            } else {
                None
            }
        }
    }

    /// A minimal, entirely synthetic "ELF-shaped" payload — never real
    /// firmware bytes. Sized to a whole number of AES blocks so it can be
    /// used directly as encrypted-segment ciphertext-length input too.
    fn synthetic_elf_payload() -> Vec<u8> {
        let mut elf = vec![0x7F, b'E', b'L', b'F', 2, 1, 1, 0];
        elf.extend(std::iter::repeat_n(0xABu8, 24)); // pad to 32 bytes (2 AES blocks)
        elf
    }

    /// Build a SELF header (32 bytes) + entry table (`entries.len()` * 32
    /// bytes), followed by the given segment payloads laid out
    /// contiguously right after the entry table. Returns the full buffer.
    fn build_self(entry_props: &[u64], segments: &[&[u8]]) -> Vec<u8> {
        assert_eq!(entry_props.len(), segments.len());
        let header_size = SELF_HEADER_SIZE + entry_props.len() * SELF_ENTRY_SIZE;

        let mut buf = vec![0u8; header_size];
        buf[0..4].copy_from_slice(&SELF_MAGIC.to_le_bytes());
        buf[4] = 1; // version
        buf[5] = 0; // mode
        buf[6] = 1; // endian
        buf[7] = 0; // attributes
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // key_type
        buf[12..14].copy_from_slice(&(header_size as u16).to_le_bytes());
        buf[14..16].copy_from_slice(&0u16.to_le_bytes()); // meta_size
        buf[24..26].copy_from_slice(&(entry_props.len() as u16).to_le_bytes());
        buf[26..28].copy_from_slice(&0u16.to_le_bytes()); // flags

        let mut offset = header_size as u64;
        for (i, (&properties, &seg)) in entry_props.iter().zip(segments.iter()).enumerate() {
            let base = SELF_HEADER_SIZE + i * SELF_ENTRY_SIZE;
            buf[base..base + 8].copy_from_slice(&properties.to_le_bytes());
            buf[base + 8..base + 16].copy_from_slice(&offset.to_le_bytes());
            buf[base + 16..base + 24].copy_from_slice(&(seg.len() as u64).to_le_bytes());
            buf[base + 24..base + 32].copy_from_slice(&(seg.len() as u64).to_le_bytes());
            offset += seg.len() as u64;
        }

        for seg in segments {
            buf.extend_from_slice(seg);
        }

        let file_size = buf.len() as u64;
        buf[16..24].copy_from_slice(&file_size.to_le_bytes());

        buf
    }

    #[test]
    fn plaintext_self_passes_through() {
        let elf = synthetic_elf_payload();
        // properties = 0: not encrypted, segment_index = 0.
        let self_data = build_self(&[0], &[&elf]);

        let result = decrypt_self(&self_data, &NoKeysProvider).expect("plaintext SELF decrypts");
        assert_eq!(result.elf, elf);
        assert_eq!(result.passthrough_segments, 1);
        assert_eq!(result.decrypted_segments, 0);
    }

    #[test]
    fn encrypted_ctr_segment_recovered_with_matching_key() {
        let elf = synthetic_elf_payload();
        let ciphertext = aes128_ctr_xor(TEST_KEY, TEST_IV, &elf);

        // properties: bit1 nonzero (but not 2) selects "encrypted + CTR";
        // segment_index (bits 20+) = TEST_KEY_ID so the stub provider's
        // KeyRequest.key_id lines up.
        let properties = (1u64 << 1) | (TEST_KEY_ID << 20);
        let self_data = build_self(&[properties], &[&ciphertext]);

        let result =
            decrypt_self(&self_data, &StubProvider).expect("stub provider recovers plaintext");
        assert_eq!(result.elf, elf);
        assert_eq!(result.decrypted_segments, 1);
        assert_eq!(result.passthrough_segments, 0);
    }

    #[test]
    fn encrypted_cbc_segment_recovered_with_matching_key() {
        let elf = synthetic_elf_payload();
        let ciphertext = aes128_cbc_encrypt(TEST_KEY, TEST_IV, &elf);

        // Real bit 1 = encrypted, plus the reserved fixture-only CBC selector.
        // (This previously set bit 2 — which is `signed` in the real format,
        // not an "algo == 2" enum value; that misreading is exactly what made
        // real titles' signed *plaintext* segments look CBC-encrypted.)
        let properties = 0x2u64 | PROP_FIXTURE_CBC | (TEST_KEY_ID << 20);
        let self_data = build_self(&[properties], &[&ciphertext]);

        let result =
            decrypt_self(&self_data, &StubProvider).expect("stub provider recovers plaintext");
        assert_eq!(result.elf, elf);
        assert_eq!(result.decrypted_segments, 1);
    }

    #[test]
    fn encrypted_segment_without_key_is_missing_key_not_panic() {
        let elf = synthetic_elf_payload();
        let ciphertext = aes128_ctr_xor(TEST_KEY, TEST_IV, &elf);
        let properties = (1u64 << 1) | (TEST_KEY_ID << 20);
        let self_data = build_self(&[properties], &[&ciphertext]);

        let err = decrypt_self(&self_data, &NoKeysProvider).unwrap_err();
        assert!(matches!(
            err,
            FirmwareError::MissingKey { key_id } if key_id == TEST_KEY_ID
        ));
    }

    #[test]
    fn truncated_header_does_not_panic() {
        let data = [0u8; 8]; // shorter than the 32-byte header
        let err = decrypt_self(&data, &NoKeysProvider).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedSelf(_)));
    }

    #[test]
    fn truncated_entry_table_does_not_panic() {
        let elf = synthetic_elf_payload();
        let mut self_data = build_self(&[0], &[&elf]);
        // Cut the buffer off partway through the (only) entry record, while
        // still claiming 1 entry in the header.
        self_data.truncate(SELF_HEADER_SIZE + 10);
        let err = decrypt_self(&self_data, &NoKeysProvider).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedSelf(_)));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let data = [0u8; 64];
        let err = decrypt_self(&data, &NoKeysProvider).unwrap_err();
        assert!(matches!(
            err,
            FirmwareError::Loader(LoaderError::InvalidSelfMagic(_))
        ));
    }

    #[test]
    fn absurd_entry_count_does_not_over_allocate() {
        let mut data = vec![0u8; SELF_HEADER_SIZE];
        data[0..4].copy_from_slice(&SELF_MAGIC.to_le_bytes());
        data[12..14].copy_from_slice(&(SELF_HEADER_SIZE as u16).to_le_bytes());
        data[24..26].copy_from_slice(&u16::MAX.to_le_bytes()); // num_entries = 65535
        let err = decrypt_self(&data, &NoKeysProvider).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedSelf(_)));
    }

    #[test]
    fn segment_data_out_of_bounds_does_not_panic() {
        // One entry claiming a segment far beyond the (tiny) file.
        let mut self_data = vec![0u8; SELF_HEADER_SIZE + SELF_ENTRY_SIZE];
        self_data[0..4].copy_from_slice(&SELF_MAGIC.to_le_bytes());
        let header_size = self_data.len() as u16;
        self_data[12..14].copy_from_slice(&header_size.to_le_bytes());
        self_data[24..26].copy_from_slice(&1u16.to_le_bytes()); // num_entries = 1

        let base = SELF_HEADER_SIZE;
        self_data[base..base + 8].copy_from_slice(&0u64.to_le_bytes()); // properties: plaintext
        self_data[base + 8..base + 16].copy_from_slice(&0xFFFF_FFFFu64.to_le_bytes()); // offset: absurd
        self_data[base + 16..base + 24].copy_from_slice(&16u64.to_le_bytes()); // compressed_size
        self_data[base + 24..base + 32].copy_from_slice(&16u64.to_le_bytes()); // uncompressed_size

        let err = decrypt_self(&self_data, &NoKeysProvider).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedSelf(_)));
    }
}
