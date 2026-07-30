//! The `memory:` pseudo-file scheme — a read-only file whose bytes live in
//! guest memory rather than on the host filesystem.
//!
//! # Why this exists
//!
//! Scaleform/GFx (the UI middleware in GTA V and many other titles) can hand
//! its file layer a buffer it has *already* loaded instead of a path, spelling
//! the handle as a URI:
//!
//! ```text
//! memory:$1559e00000,198674,0:00002_font_lib_efigs_ps5.gfx
//! memory:$<hex guest address>,<decimal byte length>,<decimal flags>:<display name>
//! ```
//!
//! That is not a host path and never was. Before this module the string fell
//! through to [`super::resolve_path`], whose per-segment sandbox check
//! correctly refuses any segment containing `:` (a Windows drive or
//! alternate-data-stream qualifier) — measured on GTA V as
//!
//! ```text
//! WARN raeen_kernel::filesystem: VFS resolve: refusing drive-qualified/absolute
//! segment in guest path 'memory:$1559e00000,198674,0:00002_font_lib_efigs_ps5.gfx'
//! ```
//!
//! so `open` returned `ENOENT` and one guest thread burned a full core retrying
//! the same five UI assets (149 attempts in 25 s). The drive-qualifier check is
//! doing exactly the right thing and is left untouched; this scheme is simply
//! routed off **before** host-path resolution ever sees it.
//!
//! # Design notes and prior art
//!
//! No emulator in `reference/` implements this — greps for `memory:`,
//! `Scaleform`, `GFx`, and `MemoryFile` across `shadps4`, `kytyps5`, `kyty`,
//! and `sharpemu` return nothing. Three *patterns* were taken from reading
//! them; no code was ported, so `THIRD_PARTY_NOTICES.md` needs no new entry:
//!
//! * **shadPS4** (`src/core/libraries/kernel/file_system.cpp:119`) branches on
//!   `path.starts_with("/dev/")` *after* flag validation and *before*
//!   `GetHostPath`, handing the full path to a device factory. The same
//!   ordering is used here — see [`super::VirtualFileSystem::open`].
//! * **sharpemu** (`KernelMemoryCompatExports.ResolveGuestPath`) makes a prefix
//!   match *authoritative*: once a prefix claims a path, a denial must not fall
//!   through to another resolver. Hence [`parse`] returns a **named** error and
//!   `open` refuses, rather than letting a malformed URI degrade into a host
//!   filesystem probe.
//! * **KytyPS5** (`src/kernel/memory.cpp:869`, `TryReadBacking`) validates and
//!   copies in one pass so there is no check-then-read window. [`GuestByteSource`]
//!   is specified the same way, and the range is *re*-validated on every read
//!   rather than trusted from `open` time.
//!
//! Sony's own firmware models this the same way: `ps4libdoc`'s symbol list
//! carries `AbstractStorage::MemfileContent::CreateInstance(std::string, void*,
//! unsigned long)` — display name, pointer, length, exactly the three fields
//! this URI carries.
//!
//! # What is deliberately NOT done
//!
//! Writes. A `memory:` handle is read-only: the guest owns that buffer and
//! already has a pointer to it, so a write through the file layer would be a
//! second, aliasing path to memory the title can just store to directly.

use std::borrow::Cow;

use raeen_core::blockers::{self, BlockerCategory};

/// URI prefix that claims a path for this scheme.
///
/// Matched ASCII-case-insensitively. Claiming is deliberately based on the
/// scheme token alone (not `memory:$`): a path that begins `memory:` but is
/// malformed must produce a *named* refusal, never silently degrade into a host
/// filesystem probe. No mount prefix in this VFS begins with `memory:`, and any
/// such path was already refused before this module existed, so claiming it
/// weakens nothing.
pub const MEMORY_SCHEME: &str = "memory:";

/// Largest declared length this scheme will serve, matching `raeen-hle`'s
/// `MAX_HLE_BULK_BYTES`. The five URIs GTA V requests are 9 KiB – 194 KiB; a
/// declared length above this is a mis-parse or a hostile value, not a UI asset.
pub const MAX_MEMORY_FILE_LEN: u64 = 256 << 20;

/// A parsed `memory:` URI: a guest byte range plus the name the title used to
/// spell it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryUri {
    /// Guest address of the first byte.
    pub addr: u64,
    /// Declared length in bytes. This is the file's size for `lseek(SEEK_END)`
    /// and `fstat`; never re-derived from anything else.
    pub len: u64,
    /// Opaque flags field. Recorded for diagnostics; no bit is interpreted,
    /// because no observed value is non-zero and guessing would be a stub
    /// pretending to be a feature.
    pub flags: u64,
    /// Display name after the final field separator. May contain `:` and any
    /// other byte the guest chose; used only for logging.
    pub display_name: String,
}

impl MemoryUri {
    /// End of the range, or `None` if `addr + len` wraps.
    #[must_use]
    pub fn end(&self) -> Option<u64> {
        self.addr.checked_add(self.len)
    }
}

/// Why a `memory:` URI was refused. Every variant has a stable
/// [`name`](MemoryUriError::name) used as the blocker key, so a refusal is
/// greppable and countable rather than an anonymous `ENOENT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUriError {
    /// The path does not begin with [`MEMORY_SCHEME`] — not this scheme's
    /// business. Callers use this to fall through to host resolution.
    NotMemoryScheme,
    /// No `$` sigil introducing the address field.
    MissingAddressSigil,
    /// `$` present but no digits followed it.
    EmptyAddress,
    /// The address field is not valid hexadecimal.
    NonHexAddress,
    /// A null guest address can never be a loaded asset.
    NullAddress,
    /// No `,` terminating the address field.
    MissingLengthField,
    /// The length field is not a valid decimal number.
    NonDecimalLength,
    /// A zero-length file would make every read return EOF; the title asked
    /// for content and would loop.
    ZeroLength,
    /// The declared length exceeds [`MAX_MEMORY_FILE_LEN`].
    LengthTooLarge,
    /// No `,` terminating the length field.
    MissingFlagsField,
    /// The flags field is not a valid decimal number.
    NonDecimalFlags,
    /// No `:` terminating the flags field, so there is no display name.
    MissingDisplayName,
    /// The display name after the final `:` is empty.
    EmptyDisplayName,
    /// `addr + len` overflows the 64-bit address space.
    RangeOverflow,
}

impl MemoryUriError {
    /// Stable, greppable identifier. Used as the blocker key so occurrences
    /// aggregate by cause.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::NotMemoryScheme => "not-memory-scheme",
            Self::MissingAddressSigil => "missing-address-sigil",
            Self::EmptyAddress => "empty-address",
            Self::NonHexAddress => "non-hex-address",
            Self::NullAddress => "null-address",
            Self::MissingLengthField => "missing-length-field",
            Self::NonDecimalLength => "non-decimal-length",
            Self::ZeroLength => "zero-length",
            Self::LengthTooLarge => "length-too-large",
            Self::MissingFlagsField => "missing-flags-field",
            Self::NonDecimalFlags => "non-decimal-flags",
            Self::MissingDisplayName => "missing-display-name",
            Self::EmptyDisplayName => "empty-display-name",
            Self::RangeOverflow => "range-overflow",
        }
    }
}

impl std::fmt::Display for MemoryUriError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Whether `path` belongs to this scheme, and so must never reach host-path
/// resolution.
/// Compared over **bytes**, not a `&str` slice: guest paths arrive through
/// `String::from_utf8_lossy`, so a title can hand over a path whose first
/// characters are multi-byte. `&path[..7]` would then panic with "byte index 7 is
/// not a char boundary" — a guest-triggerable crash in the file layer.
#[must_use]
pub fn claims(path: &str) -> bool {
    path.as_bytes()
        .get(..MEMORY_SCHEME.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(MEMORY_SCHEME.as_bytes()))
}

/// Parse `memory:$<hex addr>,<dec len>,<dec flags>:<display name>`.
///
/// Strict by construction, and positional rather than colon-splitting: the
/// display name may itself contain `:` (sharpemu learned the same lesson about
/// `app0:/…` — a colon is not a reliable scheme delimiter), so only the *first*
/// `:` after the flags field terminates it.
pub fn parse(path: &str) -> Result<MemoryUri, MemoryUriError> {
    if !claims(path) {
        return Err(MemoryUriError::NotMemoryScheme);
    }
    // `claims` proved the first `MEMORY_SCHEME.len()` bytes are ASCII, so this
    // byte index is a char boundary and the slice cannot panic.
    let body = &path[MEMORY_SCHEME.len()..];
    let body = body
        .strip_prefix('$')
        .ok_or(MemoryUriError::MissingAddressSigil)?;

    let (addr_text, rest) = body
        .split_once(',')
        .ok_or(MemoryUriError::MissingLengthField)?;
    let (len_text, rest) = rest
        .split_once(',')
        .ok_or(MemoryUriError::MissingFlagsField)?;
    let (flags_text, display_name) = rest
        .split_once(':')
        .ok_or(MemoryUriError::MissingDisplayName)?;

    if addr_text.is_empty() {
        return Err(MemoryUriError::EmptyAddress);
    }
    // `from_str_radix` accepts a leading `+`/`-`; a guest address has neither.
    if !addr_text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(MemoryUriError::NonHexAddress);
    }
    let addr = u64::from_str_radix(addr_text, 16).map_err(|_| MemoryUriError::NonHexAddress)?;
    if addr == 0 {
        return Err(MemoryUriError::NullAddress);
    }

    if len_text.is_empty() || !len_text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(MemoryUriError::NonDecimalLength);
    }
    let len: u64 = len_text
        .parse()
        .map_err(|_| MemoryUriError::NonDecimalLength)?;
    if len == 0 {
        return Err(MemoryUriError::ZeroLength);
    }
    if len > MAX_MEMORY_FILE_LEN {
        return Err(MemoryUriError::LengthTooLarge);
    }

    if flags_text.is_empty() || !flags_text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(MemoryUriError::NonDecimalFlags);
    }
    let flags: u64 = flags_text
        .parse()
        .map_err(|_| MemoryUriError::NonDecimalFlags)?;

    if display_name.is_empty() {
        return Err(MemoryUriError::EmptyDisplayName);
    }

    let uri = MemoryUri {
        addr,
        len,
        flags,
        display_name: display_name.to_string(),
    };
    if uri.end().is_none() {
        return Err(MemoryUriError::RangeOverflow);
    }
    Ok(uri)
}

/// Read-only access to the guest address space, for this scheme only.
///
/// `raeen-kernel` sits *below* `raeen-hle` and `raeen-runtime` in the
/// dependency graph, so it cannot name `GuestMemory` or `GuestArena`. This is
/// the narrow seam the runtime installs (see
/// [`super::VirtualFileSystem::set_guest_byte_source`]); it is held as a
/// [`std::sync::Weak`] so a finished process's arena is not kept alive by the
/// VFS.
///
/// # Contract
///
/// Both methods are **fail-closed**: an implementation must return `false`,
/// having read nothing the caller may rely on, unless every byte of the range
/// is mapped and readable in the guest address space. This is the boundary
/// where a guest-supplied pointer stops being an integer, so a bug here is a
/// *host* memory-safety bug rather than a wrong pixel.
pub trait GuestByteSource: Send + Sync {
    /// Copy exactly `out.len()` bytes starting at guest address `addr`.
    ///
    /// Returns `false` if `[addr, addr + out.len())` is not entirely mapped and
    /// readable. Validation and copy should be one pass (KytyPS5's
    /// `TryReadBacking` shape) so no window exists between them.
    fn read_guest_bytes(&self, addr: u64, out: &mut [u8]) -> bool;

    /// Whether `[addr, addr + len)` is entirely mapped and readable, without
    /// copying.
    ///
    /// The default probes one byte per 4 KiB page plus the last byte, which is
    /// correct but coarse; a backend with an authoritative address map should
    /// override it.
    fn guest_range_readable(&self, addr: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some(last) = addr.checked_add(len).and_then(|end| end.checked_sub(1)) else {
            return false;
        };
        let mut probe = [0u8; 1];
        let mut at = addr;
        loop {
            if !self.read_guest_bytes(at, &mut probe) {
                return false;
            }
            if at == last {
                return true;
            }
            at = at.saturating_add(0x1000).min(last);
        }
    }
}

/// Record a named, counted refusal of a `memory:` request.
///
/// Interning emits exactly one `warn!` per distinct (reason, address) pair, so
/// a guest retry loop is visible once rather than 149 times, and the count is
/// available in the blocker digest.
pub(crate) fn refuse(reason: &'static str, addr: u64, detail: impl FnOnce() -> String) {
    blockers::record(
        BlockerCategory::VfsMiss,
        Cow::Owned(format!("memory-scheme:{reason}")),
        addr,
        detail,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five URIs GTA V was measured requesting (addresses and lengths as
    /// logged; no game bytes involved).
    const MEASURED: [&str; 5] = [
        "memory:$1085a08000,9156,0:00005_initial_interactive_screen_ps5.gfx",
        "memory:$1546320000,110250,0:00010_game_stream.gfx",
        "memory:$1546320000,63305,0:00011_generic_instructional_buttons.gfx",
        "memory:$1559e00000,198674,0:00002_font_lib_efigs_ps5.gfx",
        "memory:$1559ec0000,114275,0:00015_loadingscreen_startup.gfx",
    ];

    #[test]
    fn the_five_measured_gta_v_uris_parse_to_their_declared_address_and_length() {
        let expected: [(u64, u64, u64, &str); 5] = [
            (
                0x10_85A0_8000,
                9156,
                0,
                "00005_initial_interactive_screen_ps5.gfx",
            ),
            (0x15_4632_0000, 110_250, 0, "00010_game_stream.gfx"),
            (
                0x15_4632_0000,
                63_305,
                0,
                "00011_generic_instructional_buttons.gfx",
            ),
            (0x15_59E0_0000, 198_674, 0, "00002_font_lib_efigs_ps5.gfx"),
            (
                0x15_59EC_0000,
                114_275,
                0,
                "00015_loadingscreen_startup.gfx",
            ),
        ];
        for (uri, (addr, len, flags, name)) in MEASURED.into_iter().zip(expected) {
            let parsed = parse(uri).unwrap_or_else(|e| panic!("{uri} must parse, got {e}"));
            assert_eq!(parsed.addr, addr, "{uri} address");
            assert_eq!(parsed.len, len, "{uri} length");
            assert_eq!(parsed.flags, flags, "{uri} flags");
            assert_eq!(parsed.display_name, name, "{uri} display name");
            assert!(claims(uri), "{uri} must be claimed by the scheme");
        }
    }

    #[test]
    fn every_malformed_uri_is_refused_by_its_own_name_and_never_panics() {
        let cases: [(&str, MemoryUriError); 14] = [
            ("/app0/real.gfx", MemoryUriError::NotMemoryScheme),
            ("memory", MemoryUriError::NotMemoryScheme),
            ("memory:1000,4,0:x", MemoryUriError::MissingAddressSigil),
            ("memory:$,4,0:x", MemoryUriError::EmptyAddress),
            ("memory:$zzzz,4,0:x", MemoryUriError::NonHexAddress),
            ("memory:$-10,4,0:x", MemoryUriError::NonHexAddress),
            ("memory:$0,4,0:x", MemoryUriError::NullAddress),
            ("memory:$1000", MemoryUriError::MissingLengthField),
            ("memory:$1000,4", MemoryUriError::MissingFlagsField),
            ("memory:$1000,x,0:n", MemoryUriError::NonDecimalLength),
            ("memory:$1000,,0:n", MemoryUriError::NonDecimalLength),
            ("memory:$1000,0,0:n", MemoryUriError::ZeroLength),
            ("memory:$1000,4,z:n", MemoryUriError::NonDecimalFlags),
            ("memory:$1000,4,0", MemoryUriError::MissingDisplayName),
        ];
        for (uri, expected) in cases {
            assert_eq!(
                parse(uri).unwrap_err(),
                expected,
                "{uri} must be refused as {}",
                expected.name()
            );
        }
        // Named separately: these two are size/overflow limits, not shape.
        assert_eq!(
            parse("memory:$1000,4,0:").unwrap_err(),
            MemoryUriError::EmptyDisplayName
        );
        assert_eq!(
            parse(&format!("memory:$1000,{},0:n", MAX_MEMORY_FILE_LEN + 1)).unwrap_err(),
            MemoryUriError::LengthTooLarge
        );
        assert_eq!(
            parse("memory:$ffffffffffffffff,4096,0:n").unwrap_err(),
            MemoryUriError::RangeOverflow
        );
        // A length that does not fit u64 at all is a decimal-parse refusal, not
        // a panic.
        assert_eq!(
            parse("memory:$1000,99999999999999999999999,0:n").unwrap_err(),
            MemoryUriError::NonDecimalLength
        );
    }

    #[test]
    fn a_display_name_may_contain_colons_and_commas_because_only_the_first_three_fields_are_positional()
     {
        let uri = parse("memory:$1000,16,0:C:\\weird,name:with:colons.gfx").expect("parses");
        assert_eq!(uri.addr, 0x1000);
        assert_eq!(uri.len, 16);
        assert_eq!(uri.display_name, "C:\\weird,name:with:colons.gfx");
    }

    #[test]
    fn the_scheme_token_is_case_insensitive_but_nothing_else_is_relaxed() {
        assert!(claims("MEMORY:$1000,4,0:n"));
        assert!(claims("Memory:$1000,4,0:n"));
        assert_eq!(parse("MEMORY:$1000,4,0:n").unwrap().addr, 0x1000);
        // Not the scheme: no accidental claim of an ordinary guest path.
        assert!(!claims("/app0/memory/asset.gfx"));
        assert!(!claims("memor:$1000,4,0:n"));
        assert!(!claims(""));
    }

    #[test]
    fn a_multibyte_guest_path_is_classified_without_panicking_on_a_char_boundary() {
        // Guest paths arrive via `String::from_utf8_lossy`, so a title can hand
        // over multi-byte characters. Byte-indexing a `&str` prefix would panic
        // ("byte index 7 is not a char boundary") — a guest-triggerable crash in
        // the file layer, which is why `claims` compares bytes.
        for path in [
            "мемory:$1000,4,0:n",        // Cyrillic: 2 bytes per letter
            "\u{10FFFF}\u{10FFFF}x",     // 4-byte chars straddling index 7
            "mem\u{FFFD}ry:$1000,4,0:n", // the lossy replacement char itself
            "/app0/日本語/asset.gfx",    // ordinary path, multi-byte segment
            "me",                        // shorter than the scheme token
            "\u{00E9}",                  // 2 bytes total
        ] {
            assert!(!claims(path), "{path:?} must not be claimed");
            assert_eq!(parse(path).unwrap_err(), MemoryUriError::NotMemoryScheme);
        }
        // A multi-byte display name is fine — it is opaque.
        let uri = parse("memory:$1000,4,0:日本語.gfx").expect("multi-byte name parses");
        assert_eq!(uri.display_name, "日本語.gfx");
    }

    #[test]
    fn the_default_range_probe_is_fail_closed_across_page_boundaries() {
        /// Readable only inside `[base, base+span)`.
        struct Window {
            base: u64,
            span: u64,
        }
        impl GuestByteSource for Window {
            fn read_guest_bytes(&self, addr: u64, out: &mut [u8]) -> bool {
                let Some(end) = addr.checked_add(out.len() as u64) else {
                    return false;
                };
                if addr < self.base || end > self.base + self.span {
                    return false;
                }
                out.fill(0x5A);
                true
            }
        }
        let window = Window {
            base: 0x1_0000,
            span: 0x3000,
        };
        assert!(window.guest_range_readable(0x1_0000, 0x3000), "exact fit");
        assert!(
            window.guest_range_readable(0x1_0FFF, 0x2001),
            "unaligned fit"
        );
        assert!(
            !window.guest_range_readable(0x1_0000, 0x3001),
            "one byte past the window must fail closed"
        );
        assert!(
            !window.guest_range_readable(0xFFFF, 0x10),
            "straddling the low edge must fail closed"
        );
        assert!(
            !window.guest_range_readable(u64::MAX - 1, 0x10),
            "an overflowing range must fail closed"
        );
        assert!(
            window.guest_range_readable(0x1_0000, 0),
            "empty is trivially readable"
        );
    }
}
