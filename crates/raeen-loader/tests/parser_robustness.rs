//! Property tests (proptest): the binary parsers consume user-supplied,
//! potentially hostile files, so for ANY input bytes they must return an
//! error — never panic, never overflow an index. These generators shrink a
//! failure to a minimal counterexample automatically.

use proptest::prelude::*;
use raeen_loader::elf::parse_elf;
use raeen_loader::pkg::parse_pkg_header;

proptest! {
    /// Arbitrary bytes through the ELF parser: any outcome but a panic.
    #[test]
    fn parse_elf_never_panics_on_arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..2048)) {
        let _ = parse_elf(&data);
    }

    /// Bytes that *start* like an ELF (magic + arbitrary header tail) probe
    /// deeper parser paths than pure noise does.
    #[test]
    fn parse_elf_never_panics_on_elf_prefixed_bytes(tail in proptest::collection::vec(any::<u8>(), 0..1024)) {
        let mut data = b"\x7fELF".to_vec();
        data.extend(tail);
        let _ = parse_elf(&data);
    }

    /// Arbitrary bytes through the PKG header parser.
    #[test]
    fn parse_pkg_header_never_panics_on_arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..512)) {
        let _ = parse_pkg_header(&data);
    }
}
