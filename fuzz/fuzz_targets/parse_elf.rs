//! Fuzz target: the ELF parser against arbitrary bytes. The parser consumes
//! user-supplied files (homebrew, decrypted modules), so any input must
//! produce `Ok`/`Err` — never a panic, index overflow, or OOM.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = raeen_loader::elf::parse_elf(data);
});
