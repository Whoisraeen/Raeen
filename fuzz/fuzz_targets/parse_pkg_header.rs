//! Fuzz target: the PKG header parser against arbitrary bytes (user-supplied
//! package files are untrusted input).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = raeen_loader::pkg::parse_pkg_header(data);
});
