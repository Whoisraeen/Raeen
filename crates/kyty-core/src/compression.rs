//! Port of Kyty's `Kyty::Core::Compression`
//! (`reference/kyty/source/include/Kyty/Core/Compression.h`,
//! `reference/kyty/source/lib/Core/src/Compression.cpp`).
//!
//! # Scope of this pass
//!
//! The C++ header bundles two unrelated things: (1) a set of free
//! `Compress*`/`Decompress*` codec functions operating on byte buffers, and
//! (2) `ZipReader`/`ZipWriter`/`ZipFileStat`, a `miniz`-backed **archive**
//! (container-format) API that reads/writes whole `.zip` files through
//! `Kyty::Core::File`. `File` has not been ported yet (it is not in this
//! crate's "already ported" set), so `ZipReader`/`ZipWriter`/`ZipFileStat`
//! are **not** ported in this pass — they are pure file-I/O plumbing on top
//! of the codecs below and can be added once `File` lands, without touching
//! anything here. This module ports the codec half only: `CompressZstd`/
//! `CompressLzma`/`CompressZip`/`CompressLzf` and their `Decompress*`
//! counterparts (including the `*Str` variants).
//!
//! # Overload collapsing
//!
//! Every codec in the C++ header is overloaded three ways:
//! `(const uint8_t* buf, uint32_t length, ...)`, `(const ByteBuffer&, ...)`,
//! and `(const String&, ...)` — the first two overloads always just forward
//! to the raw-pointer form. Rust has no overloading, but
//! [`crate::byte_buffer::ByteBuffer`] derefs to `Vec<u8>` which derefs to
//! `[u8]`, so a single function taking `&[u8]` already accepts a plain slice
//! *or* `&ByteBuffer` via (two-hop) deref coercion — no separate
//! `compress_zstd_buf`-style wrapper is needed. Only the `String` overload
//! gets its own `_str`-suffixed function, one per codec.
//!
//! # `String` interop
//!
//! `Kyty::Core::String` (the Unicode string type) is not part of this
//! crate's already-ported set. Per the port conventions, the `*Str`
//! functions here use [`crate::string8::String8`] (already ported) for the
//! UTF-8 byte content instead: `CompressX(const String& str)` called
//! `str.utf8_str()` to get UTF-8 bytes before compressing, and `Decompress*
//! Str` reconstructs a `String` from a NUL-terminated decompressed buffer via
//! `String::FromUtf8`. `String8` already *is* a UTF-8-capable byte string, so
//! `compress_*_str` takes `&String8` directly (skipping the
//! Unicode-round-trip step Kyty's own callers went through) and
//! `decompress_*_str` returns a `String8` (skipping the reverse step).
//!
//! Kyty's own `CompressX(String)` / `DecompressXStr` pair is **not** a true
//! round trip: `utf8_str()` never appends a trailing NUL, but
//! `DecompressXStr` unconditionally asserts the *last byte of the
//! decompressed data* is `0` and treats the content as a NUL-terminated C
//! string (`String::FromUtf8(const char*)`). Compressing a `String8` with
//! `compress_x_str` and decompressing the result with `decompress_x_str`
//! therefore does **not** round-trip here either (by design — this mirrors
//! Kyty exactly, not a bug in this port): `decompress_x_str` is the
//! faithful counterpart to data that was compressed from an explicitly
//! NUL-terminated byte buffer, not from `compress_x_str`'s own output. See
//! the module tests for both the intra-crate round trip (`compress_x` /
//! `decompress_x`) and this documented asymmetry.
//!
//! # Codec -> crate mapping
//!
//! - **Zstd**: `ZSTD_compress`/`ZSTD_decompressStream` -> the `zstd` crate
//!   (already a workspace dependency), via `zstd::stream::encode_all`/
//!   `decode_all`.
//! - **Lzma**: Kyty wraps the (external, C) LZMA SDK behind hand-rolled
//!   `ISeqInStream`/`ISeqOutStream` callback structs, emitting a classic
//!   "LZMA alone" stream: 5 bytes of encoder properties + 8 bytes of
//!   little-endian uncompressed size, then the raw LZMA-compressed data (no
//!   end-of-stream marker; the decoder is told the exact unpacked size up
//!   front). Reimplementing the LZMA algorithm itself from scratch is out of
//!   scope for a faithful *port* (it isn't Kyty's own algorithm, unlike
//!   LZF below); this maps to the `lzma-rs` crate (pure-Rust, safe,
//!   `lzma_compress`/`lzma_decompress`), which produces/consumes the same
//!   classic LZMA-alone container shape. Needs `lzma-rs` added to the
//!   workspace (see `needs_crates`).
//! - **Zip**: despite the name, this is *not* the `.zip` archive format —
//!   `mz_deflateInit(level)` calls `mz_deflateInit2` with
//!   `window_bits = MZ_DEFAULT_WINDOW_BITS` (positive `15`), which per
//!   `miniz.h` wraps the deflate stream with a zlib header + Adler-32
//!   footer, i.e. plain **zlib** framing. Maps to the `flate2` crate's
//!   `write::ZlibEncoder`/`read::ZlibDecoder`, per the port conventions'
//!   explicit "wraps zlib/deflate/gzip -> `flate2`" guidance. Needs
//!   `flate2` added to the workspace (see `needs_crates`).
//! - **Lzf**: Kyty's *own* hand-rolled LZ77-family codec (`lzf_compress`/
//!   `lzf_decompress` in `Compression.cpp`, not an external library) written
//!   in raw-pointer C. Per the port conventions ("do NOT transliterate
//!   manual-memory/raw-pointer code into unsafe Rust"), this is reimplemented
//!   in the private [`lzf`] submodule using bounds-checked slice indexing
//!   (`usize`/`i64` offsets in place of `uint8_t*` pointers) instead of
//!   pointer arithmetic — same algorithm and wire format, zero `unsafe`.
//!
//! # Documented safety-related divergences
//!
//! - `CompressLzma`/`DecompressLzma` fault (`EXIT_IF`) on an empty input in
//!   the original; `compress_lzma`/`decompress_lzma` preserve that via
//!   `exit_if!`.
//! - Kyty's `lzf_compress`/`lzf_decompress` unconditionally read/write one
//!   byte before checking the input is non-empty, which for a *zero-length*
//!   input is an (unexercised in practice) one-byte-out-of-bounds C++ access.
//!   `compress_lzf`/`decompress_lzf` special-case empty input to return an
//!   empty buffer instead of reproducing that access.

use crate::byte_buffer::ByteBuffer;
use crate::string8::String8;
use std::io::{Cursor, Read, Write};

/// `Kyty::Core::ZipCompressLevel` (`using ZipCompressLevel = int;`).
pub type ZipCompressLevel = i32;
/// `Kyty::Core::ZstdCompressLevel` (`using ZstdCompressLevel = int;`).
pub type ZstdCompressLevel = i32;

pub const ZIP_NO_COMPRESSION: ZipCompressLevel = 0;
pub const ZIP_BEST_SPEED: ZipCompressLevel = 1;
pub const ZIP_DEFAULT_LEVEL: ZipCompressLevel = 6;
pub const ZIP_BEST_COMPRESSION: ZipCompressLevel = 9;

pub const ZSTD_BEST_SPEED: ZstdCompressLevel = 1;
pub const ZSTD_DEFAULT_LEVEL: ZstdCompressLevel = 3;
pub const ZSTD_BEST_COMPRESSION: ZstdCompressLevel = 22;

/// Shared tail of `Decompress{Zstd,Lzma,Zip,Lzf}Str`: all four have the
/// identical structure `EXIT_IF(last byte != 0); return
/// String::FromUtf8(...)`. `String::FromUtf8(const char*)` reads a
/// NUL-terminated C string — i.e. up to (and excluding) the first NUL byte —
/// mirrored here by locating that byte and slicing up to it. See the module
/// doc comment for the `compress_x_str`/`decompress_x_str` asymmetry this
/// assumes (the caller is expected to supply data that was compressed from
/// an explicitly NUL-terminated buffer).
fn nul_terminated_to_string8(data: ByteBuffer) -> String8 {
    let bytes = data.get_data();
    crate::exit_if!(bytes.is_empty() || *bytes.last().unwrap() != 0);
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String8::from_bytes(&bytes[..end])
}

// ---------------------------------------------------------------------
// Zstd
// ---------------------------------------------------------------------

/// `CompressZstd(const uint8_t* buf, uint32_t length, int level =
/// ZSTD_DEFAULT_LEVEL)` / `CompressZstd(const ByteBuffer&, int level)`
/// (the latter via deref coercion; see module doc comment).
pub fn compress_zstd(buf: &[u8], level: ZstdCompressLevel) -> ByteBuffer {
    let out = zstd::stream::encode_all(buf, level).expect("kyty: zstd compression failed");
    ByteBuffer::from(out)
}

/// `CompressZstd(const String& str, int level)`.
pub fn compress_zstd_str(s: &String8, level: ZstdCompressLevel) -> ByteBuffer {
    compress_zstd(s.as_bytes(), level)
}

/// `DecompressZstd(const uint8_t* buf, uint32_t length)` / `DecompressZstd
/// (const ByteBuffer&)`.
pub fn decompress_zstd(buf: &[u8]) -> ByteBuffer {
    let out = zstd::stream::decode_all(buf).expect("kyty: zstd decompression failed");
    ByteBuffer::from(out)
}

/// `DecompressZstdStr(const uint8_t*, uint32_t)` / `DecompressZstdStr(const
/// ByteBuffer&)`.
pub fn decompress_zstd_str(buf: &[u8]) -> String8 {
    nul_terminated_to_string8(decompress_zstd(buf))
}

// ---------------------------------------------------------------------
// Lzma
// ---------------------------------------------------------------------

/// `CompressLzma(const uint8_t* buf, uint32_t length)` / `CompressLzma(const
/// ByteBuffer&)`. Faithfully preserves Kyty's `EXIT_IF(!buf); EXIT_IF(length
/// == 0);` — compressing empty input is not supported.
pub fn compress_lzma(buf: &[u8]) -> ByteBuffer {
    crate::exit_if!(buf.is_empty());
    let mut input = Cursor::new(buf);
    let mut output = Vec::new();
    lzma_rs::lzma_compress(&mut input, &mut output).expect("kyty: LZMA compression failed");
    ByteBuffer::from(output)
}

/// `CompressLzma(const String& str)`.
pub fn compress_lzma_str(s: &String8) -> ByteBuffer {
    compress_lzma(s.as_bytes())
}

/// `DecompressLzma(const uint8_t* buf, uint32_t length)` / `DecompressLzma
/// (const ByteBuffer&)`. Faithfully preserves Kyty's `EXIT_IF(!buf);
/// EXIT_IF(length == 0);`.
pub fn decompress_lzma(buf: &[u8]) -> ByteBuffer {
    crate::exit_if!(buf.is_empty());
    let mut input = Cursor::new(buf);
    let mut output = Vec::new();
    lzma_rs::lzma_decompress(&mut input, &mut output).expect("kyty: LZMA decompression failed");
    ByteBuffer::from(output)
}

/// `DecompressLzmaStr(const uint8_t*, uint32_t)` / `DecompressLzmaStr(const
/// ByteBuffer&)`.
pub fn decompress_lzma_str(buf: &[u8]) -> String8 {
    nul_terminated_to_string8(decompress_lzma(buf))
}

// ---------------------------------------------------------------------
// Zip (zlib framing — see module doc comment)
// ---------------------------------------------------------------------

/// `CompressZip(const uint8_t* buf, uint32_t length, ZipCompressLevel level
/// = ZIP_DEFAULT_LEVEL)` / `CompressZip(const ByteBuffer&, ZipCompressLevel)`.
pub fn compress_zip(buf: &[u8], level: ZipCompressLevel) -> ByteBuffer {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level as u32));
    encoder.write_all(buf).expect("kyty: zlib compression write failed");
    let out = encoder.finish().expect("kyty: zlib compression finish failed");
    ByteBuffer::from(out)
}

/// `CompressZip(const String& str, ZipCompressLevel level)`.
pub fn compress_zip_str(s: &String8, level: ZipCompressLevel) -> ByteBuffer {
    compress_zip(s.as_bytes(), level)
}

/// `DecompressZip(const uint8_t* buf, uint32_t length)` / `DecompressZip
/// (const ByteBuffer&)`.
pub fn decompress_zip(buf: &[u8]) -> ByteBuffer {
    let mut decoder = flate2::read::ZlibDecoder::new(buf);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).expect("kyty: zlib decompression failed");
    ByteBuffer::from(out)
}

/// `DecompressZipStr(const uint8_t*, uint32_t)` / `DecompressZipStr(const
/// ByteBuffer&)`.
pub fn decompress_zip_str(buf: &[u8]) -> String8 {
    nul_terminated_to_string8(decompress_zip(buf))
}

// ---------------------------------------------------------------------
// Lzf — Kyty's own hand-rolled LZ77-family codec (see `lzf` submodule)
// ---------------------------------------------------------------------

/// `CompressLzf(const uint8_t* buf, uint32_t length)` / `CompressLzf(const
/// ByteBuffer&)`. See the module doc comment for the empty-input divergence.
pub fn compress_lzf(buf: &[u8]) -> ByteBuffer {
    if buf.is_empty() {
        return ByteBuffer::new();
    }
    let estimate = lzf::calc_compressed_size(buf) as usize;
    let mut out = vec![0u8; estimate * 2];
    let actual = lzf::compress(buf, &mut out) as usize;
    crate::exit_if!(actual > out.len());
    out.truncate(actual);
    ByteBuffer::from(out)
}

/// `CompressLzf(const String& str)`.
pub fn compress_lzf_str(s: &String8) -> ByteBuffer {
    compress_lzf(s.as_bytes())
}

/// `DecompressLzf(const uint8_t* buf, uint32_t length)` / `DecompressLzf
/// (const ByteBuffer&)`. See the module doc comment for the empty-input
/// divergence.
pub fn decompress_lzf(buf: &[u8]) -> ByteBuffer {
    if buf.is_empty() {
        return ByteBuffer::new();
    }
    let size = lzf::calc_decompressed_size(buf) as usize;
    let mut out = vec![0u8; size];
    let actual = lzf::decompress(buf, &mut out);
    crate::exit_if!(actual as usize != out.len());
    ByteBuffer::from(out)
}

/// `DecompressLzfStr(const uint8_t*, uint32_t)` / `DecompressLzfStr(const
/// ByteBuffer&)`.
pub fn decompress_lzf_str(buf: &[u8]) -> String8 {
    nul_terminated_to_string8(decompress_lzf(buf))
}

/// Kyty's own hand-rolled LZF-family LZ77 codec (`lzf_calc_compressed_size`/
/// `lzf_compress`/`lzf_calc_decompressed_size`/`lzf_decompress` in
/// `Compression.cpp`). Ported 1:1 (same hashing/matching structure, same
/// control-byte wire format) with C `uint8_t*` pointers replaced by bounds-
/// checked `usize`/`i64` offsets into the caller's slices — no `unsafe`, no
/// raw pointers. See the parent module doc comment for the rationale (this
/// one codec, unlike the others, is Kyty's *own* algorithm rather than a
/// wrapped external library, so there is no crate to map it to).
mod lzf {
    const HASH_LOG: u32 = 12;
    const HASH_SIZE: usize = 1 << HASH_LOG;
    const HASH_MASK: u32 = (HASH_SIZE as u32) - 1;

    const MAX_COPY: i32 = 32;
    const MAX_LEN: i32 = 264;
    const MAX_DISTANCE: i64 = 8192;

    /// Reads two bytes at `data[idx]`/`data[idx + 1]` as a little-endian
    /// `u16` (widened to `u32`). Stands in for C's unaligned
    /// `*(const uint16_t*)ptr` read; the exact byte order doesn't affect
    /// correctness (only used for equality comparisons and as hash-table
    /// input — see module doc comment).
    #[inline]
    fn read_u16(data: &[u8], idx: usize) -> u32 {
        u32::from(data[idx]) | (u32::from(data[idx + 1]) << 8)
    }

    /// `UPDATE_HASH(uint32_t* v, const uint8_t* p)`. The C macro's first
    /// statement always overwrites `*v` outright (`(*v) = *(uint16_t*)p`),
    /// so the "input" `*v` value it also reads on the next line is really
    /// just `read_u16(p)` again — this is a pure function of `(data, p)`.
    #[inline]
    fn update_hash(data: &[u8], p: usize) -> u32 {
        let v = read_u16(data, p);
        v ^ (read_u16(data, p + 1) ^ (v >> (16 - HASH_LOG)))
    }

    /// `lzf_calc_compressed_size`.
    pub fn calc_compressed_size(input: &[u8]) -> u32 {
        let length = input.len();
        let ip_limit: i64 = length as i64 - i64::from(MAX_COPY) - 4;
        let mut htab = [0i64; HASH_SIZE];

        let mut ip: i64 = 0;
        let mut copy: i32 = 0;
        let mut opl: u32 = 1; // initial literal-run control byte

        while ip < ip_limit {
            let ip_u = ip as usize;
            let hval = update_hash(input, ip_u);
            let slot = (hval & HASH_MASK) as usize;
            let refp = htab[slot];
            htab[slot] = ip;

            if ip == refp
                || read_u16(input, refp as usize) != read_u16(input, ip_u)
                || input[refp as usize + 2] != input[ip_u + 2]
                || (ip - refp) >= MAX_DISTANCE
            {
                ip += 1;
                opl += 1;
                copy += 1;
                if copy >= MAX_COPY {
                    copy = 0;
                    opl += 1;
                }
                continue;
            }

            let anchor = ip;
            let mut len: i32 = 3;
            let mut refp2 = refp + 3;
            ip += 3;

            if ip < ip_limit - i64::from(MAX_LEN) {
                'inner: while len < MAX_LEN - 8 {
                    for _ in 0..8 {
                        let rb = input[refp2 as usize];
                        refp2 += 1;
                        let ib = input[ip as usize];
                        ip += 1;
                        if rb != ib {
                            break 'inner;
                        }
                    }
                    len += 8;
                }
                ip -= 1;
            }
            len = (ip - anchor) as i32;
            ip = anchor + i64::from(len);

            if copy != 0 {
                copy = 0;
            } else {
                opl -= 1;
            }

            len -= 2;

            opl += if len < 7 { 1 } else { 2 };
            opl += 2;

            ip -= 1;
            let hval2 = update_hash(input, ip as usize);
            htab[(hval2 & HASH_MASK) as usize] = ip;
            ip += 1;
        }

        let end = length as i64;
        while ip < end {
            ip += 1;
            opl += 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                opl += 1;
            }
        }

        if copy == 0 {
            opl -= 1;
        }

        opl
    }

    /// `lzf_compress`. `output` must be at least [`calc_compressed_size`]
    /// bytes long (the caller sizes it with headroom, as Kyty's own
    /// `CompressLzf` does).
    pub fn compress(input: &[u8], output: &mut [u8]) -> u32 {
        let length = input.len();
        let ip_limit: i64 = length as i64 - i64::from(MAX_COPY) - 4;
        let mut htab = [0i64; HASH_SIZE];

        let mut ip: i64 = 0;
        let mut op: usize = 0;
        let mut copy: i32 = 0;

        output[op] = (MAX_COPY - 1) as u8;
        op += 1;

        while ip < ip_limit {
            let ip_u = ip as usize;
            let hval = update_hash(input, ip_u);
            let slot = (hval & HASH_MASK) as usize;
            let refp = htab[slot];
            htab[slot] = ip;

            let mut distance = ip - refp;

            if ip == refp
                || read_u16(input, refp as usize) != read_u16(input, ip_u)
                || input[refp as usize + 2] != input[ip_u + 2]
                || distance >= MAX_DISTANCE
            {
                output[op] = input[ip_u];
                op += 1;
                ip += 1;
                copy += 1;
                if copy >= MAX_COPY {
                    copy = 0;
                    output[op] = (MAX_COPY - 1) as u8;
                    op += 1;
                }
                continue;
            }

            let anchor = ip;
            let mut len: i32 = 3;
            let mut refp2 = refp + 3;
            ip += 3;

            if ip < ip_limit - i64::from(MAX_LEN) {
                'inner: while len < MAX_LEN - 8 {
                    for _ in 0..8 {
                        let rb = input[refp2 as usize];
                        refp2 += 1;
                        let ib = input[ip as usize];
                        ip += 1;
                        if rb != ib {
                            break 'inner;
                        }
                    }
                    len += 8;
                }
                ip -= 1;
            }
            len = (ip - anchor) as i32;
            ip = anchor + i64::from(len);

            if copy != 0 {
                output[op - copy as usize - 1] = (copy - 1) as u8;
                copy = 0;
            } else {
                op -= 1;
            }

            len -= 2;
            distance -= 1;

            if len < 7 {
                output[op] = (((len as u32) << 5) + ((distance as u32) >> 8)) as u8;
                op += 1;
            } else {
                output[op] = ((7u32 << 5) + ((distance as u32) >> 8)) as u8;
                op += 1;
                output[op] = (len - 7) as u8;
                op += 1;
            }

            output[op] = (distance & 0xFF) as u8;
            op += 1;
            output[op] = (MAX_COPY - 1) as u8;
            op += 1;

            ip -= 1;
            let hval2 = update_hash(input, ip as usize);
            htab[(hval2 & HASH_MASK) as usize] = ip;
            ip += 1;
        }

        let end = length as i64;
        while ip < end {
            output[op] = input[ip as usize];
            op += 1;
            ip += 1;
            copy += 1;
            if copy == MAX_COPY {
                copy = 0;
                output[op] = (MAX_COPY - 1) as u8;
                op += 1;
            }
        }

        if copy != 0 {
            output[op - copy as usize - 1] = (copy - 1) as u8;
        } else {
            op -= 1;
        }

        op as u32
    }

    /// `lzf_calc_decompressed_size`. The C original reaches this result via
    /// a manually-unrolled 3-deep `if`-ladder-then-loop for the literal-run
    /// case; collapsed here to a single loop of the same total iteration
    /// count (`ctrl`) — same observable result, see the parent module's doc
    /// comment on "not byte-identical C++ idioms".
    pub fn calc_decompressed_size(input: &[u8]) -> u32 {
        let length = input.len();
        if length == 0 {
            return 0;
        }
        let ip_limit = length - 1;
        let mut ip: usize = 0;
        let mut opl: u32 = 0;

        while ip < ip_limit {
            let byte0 = input[ip];
            let ctrl = u32::from(byte0) + 1;
            let mut len = i32::from(byte0 >> 5);
            ip += 1;

            if ctrl < 33 {
                opl += ctrl;
                ip += ctrl as usize;
            } else {
                len -= 1;
                if len == 6 {
                    len += i32::from(input[ip]);
                    ip += 1;
                }
                ip += 1;
                opl += 3;
                if len > 0 {
                    opl += len as u32;
                }
            }
        }

        opl
    }

    /// `lzf_decompress`. Returns `0` on any detected overflow/corruption,
    /// matching Kyty's own error sentinel. Unlike the C original (which has
    /// no bounds checks on the *input* side and would read out of bounds /
    /// invoke UB on malformed compressed data), out-of-range input indexing
    /// here safely panics instead — an intentional, documented safety
    /// improvement (see parent module doc comment).
    pub fn decompress(input: &[u8], output: &mut [u8]) -> u32 {
        let length = input.len();
        if length == 0 {
            return 0;
        }
        let ip_limit = length - 1;
        let maxout = output.len();
        let mut ip: usize = 0;
        let mut op: usize = 0;

        while ip < ip_limit {
            let byte0 = input[ip];
            let ctrl = u32::from(byte0) + 1;
            let ofs = u32::from(byte0 & 31) << 8;
            let mut len = i32::from(byte0 >> 5);
            ip += 1;

            if ctrl < 33 {
                if op + ctrl as usize > maxout {
                    return 0;
                }
                for _ in 0..ctrl {
                    output[op] = input[ip];
                    op += 1;
                    ip += 1;
                }
            } else {
                len -= 1;

                let mut refp: i64 = op as i64 - i64::from(ofs) - 1;

                if len == 6 {
                    len += i32::from(input[ip]);
                    ip += 1;
                }

                refp -= i64::from(input[ip]);
                ip += 1;

                if op + len as usize + 3 > maxout {
                    return 0;
                }
                if refp < 0 {
                    return 0;
                }
                let mut refp = refp as usize;

                output[op] = output[refp];
                op += 1;
                refp += 1;
                output[op] = output[refp];
                op += 1;
                refp += 1;
                output[op] = output[refp];
                op += 1;
                refp += 1;

                if len > 0 {
                    for _ in 0..len {
                        output[op] = output[refp];
                        op += 1;
                        refp += 1;
                    }
                }
            }
        }

        op as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat_text(pattern: &str, times: usize) -> Vec<u8> {
        pattern.repeat(times).into_bytes()
    }

    // -------------------------------------------------------------
    // Zstd
    // -------------------------------------------------------------

    #[test]
    fn zstd_round_trip_default_level() {
        let data = b"the quick brown fox jumps over the lazy dog".repeat(20);
        let compressed = compress_zstd(&data, ZSTD_DEFAULT_LEVEL);
        assert!(compressed.size() < data.len(), "repetitive data should shrink");
        let decompressed = decompress_zstd(compressed.get_data());
        assert_eq!(decompressed.get_data(), data.as_slice());
    }

    #[test]
    fn zstd_round_trip_empty_and_extreme_levels() {
        let empty: &[u8] = b"";
        let c = compress_zstd(empty, ZSTD_BEST_SPEED);
        assert_eq!(decompress_zstd(c.get_data()).get_data(), empty);

        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let best = compress_zstd(data, ZSTD_BEST_COMPRESSION);
        assert_eq!(decompress_zstd(best.get_data()).get_data(), data);
    }

    #[test]
    fn zstd_str_and_byte_buffer_overload_via_deref() {
        let s = String8::from("hello, kyty!");
        let compressed = compress_zstd_str(&s, ZSTD_DEFAULT_LEVEL);

        // Deref coercion: `&ByteBuffer` -> `&[u8]` lets the raw-pointer-style
        // entry point double as the `ByteBuffer` overload.
        let decompressed = decompress_zstd(&compressed);
        assert_eq!(decompressed.get_data(), s.as_bytes());
    }

    #[test]
    fn zstd_decompress_str_requires_trailing_nul() {
        // compress_zstd_str does NOT append a NUL (matches Kyty's
        // `utf8_str()`), so decompressing it back through the `_str` path
        // (which requires one) is expected to fault. This documents the
        // asymmetry called out in the module doc comment.
        let s = String8::from("no nul here");
        let compressed = compress_zstd_str(&s, ZSTD_DEFAULT_LEVEL);
        let result = std::panic::catch_unwind(|| decompress_zstd_str(compressed.get_data()));
        assert!(result.is_err());

        // The faithful pairing: compress a NUL-terminated byte buffer with
        // the raw entry point, then decompress with the `_str` entry point.
        let mut with_nul = s.as_bytes().to_vec();
        with_nul.push(0);
        let compressed2 = compress_zstd(&with_nul, ZSTD_DEFAULT_LEVEL);
        assert_eq!(decompress_zstd_str(compressed2.get_data()), s);
    }

    // -------------------------------------------------------------
    // Lzma
    // -------------------------------------------------------------

    #[test]
    fn lzma_round_trip() {
        let data = repeat_text("abcabcabc123123123", 50);
        let compressed = compress_lzma(&data);
        let decompressed = decompress_lzma(compressed.get_data());
        assert_eq!(decompressed.get_data(), data.as_slice());
    }

    #[test]
    fn lzma_str_round_trip() {
        let s = String8::from("kyty lzma round trip");
        let compressed = compress_lzma_str(&s);
        let decompressed = decompress_lzma(compressed.get_data());
        assert_eq!(decompressed.get_data(), s.as_bytes());
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn lzma_compress_rejects_empty_input() {
        compress_lzma(&[]);
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn lzma_decompress_rejects_empty_input() {
        decompress_lzma(&[]);
    }

    // -------------------------------------------------------------
    // Zip (zlib)
    // -------------------------------------------------------------

    #[test]
    fn zip_round_trip_all_levels() {
        let data = b"zlib framed data zlib framed data zlib framed data".repeat(10);
        for level in [ZIP_NO_COMPRESSION, ZIP_BEST_SPEED, ZIP_DEFAULT_LEVEL, ZIP_BEST_COMPRESSION] {
            let compressed = compress_zip(&data, level);
            let decompressed = decompress_zip(compressed.get_data());
            assert_eq!(decompressed.get_data(), data.as_slice(), "level {level}");
        }
    }

    #[test]
    fn zip_has_zlib_header() {
        // A zlib stream's first two bytes form a valid CMF/FLG header (CMF
        // low nibble == 8 for deflate; the 16-bit big-endian pair is a
        // multiple of 31) — confirms this is zlib framing, not raw deflate.
        let compressed = compress_zip(b"some data to compress", ZIP_DEFAULT_LEVEL);
        let bytes = compressed.get_data();
        assert_eq!(bytes[0] & 0x0F, 8);
        let header = (u16::from(bytes[0]) << 8) | u16::from(bytes[1]);
        assert_eq!(header % 31, 0);
    }

    #[test]
    fn zip_str_round_trip() {
        let s = String8::from("zip str payload");
        let compressed = compress_zip_str(&s, ZIP_DEFAULT_LEVEL);
        assert_eq!(decompress_zip(compressed.get_data()).get_data(), s.as_bytes());
    }

    // -------------------------------------------------------------
    // Lzf
    // -------------------------------------------------------------

    #[test]
    fn lzf_round_trip_empty() {
        let c = compress_lzf(&[]);
        assert!(c.is_empty());
        let d = decompress_lzf(&[]);
        assert!(d.is_empty());
    }

    #[test]
    fn lzf_round_trip_all_literal_no_repeats() {
        // Random-looking, non-repeating bytes: forces the pure literal-run
        // path (no back-references found).
        let data: Vec<u8> = (0u32..500).map(|i| ((i.wrapping_mul(2654435761u32)) >> 24) as u8).collect();
        let compressed = compress_lzf(&data);
        let decompressed = decompress_lzf(compressed.get_data());
        assert_eq!(decompressed.get_data(), data.as_slice());
    }

    #[test]
    fn lzf_round_trip_highly_repetitive() {
        // Forces long back-reference matches (including the extended-length
        // encoding path for matches >= 7+9 bytes).
        let data = repeat_text("0123456789ABCDEF", 2000);
        let compressed = compress_lzf(&data);
        assert!(compressed.size() < data.len(), "should compress well");
        let decompressed = decompress_lzf(compressed.get_data());
        assert_eq!(decompressed.get_data(), data.as_slice());
    }

    #[test]
    fn lzf_round_trip_various_small_sizes() {
        for len in [1usize, 2, 3, 31, 32, 33, 35, 36, 37, 100] {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let compressed = compress_lzf(&data);
            let decompressed = decompress_lzf(compressed.get_data());
            assert_eq!(decompressed.get_data(), data.as_slice(), "len={len}");
        }
    }

    #[test]
    fn lzf_round_trip_long_distance_match() {
        // A match candidate far behind (near MAX_DISTANCE) plus fresh tail
        // data, exercising the distance/offset encoding.
        let mut data = vec![0xABu8; 4096];
        data.extend_from_slice(b"UNIQUE_TAIL_MARKER_SEQUENCE_1234567890");
        data.extend(std::iter::repeat_n(0xABu8, 4096));
        data.extend_from_slice(b"UNIQUE_TAIL_MARKER_SEQUENCE_1234567890");
        let compressed = compress_lzf(&data);
        let decompressed = decompress_lzf(compressed.get_data());
        assert_eq!(decompressed.get_data(), data.as_slice());
    }

    #[test]
    fn lzf_str_round_trip() {
        let s = String8::from("lzf str payload lzf str payload lzf str payload");
        let compressed = compress_lzf_str(&s);
        assert_eq!(decompress_lzf(compressed.get_data()).get_data(), s.as_bytes());
    }

    #[test]
    fn lzf_calc_compressed_size_matches_actual_output_len() {
        let data = repeat_text("mismatch-check-pattern", 37);
        let estimate = lzf::calc_compressed_size(&data);
        let mut out = vec![0u8; estimate as usize * 2];
        let actual = lzf::compress(&data, &mut out);
        assert_eq!(actual, estimate);
    }
}
