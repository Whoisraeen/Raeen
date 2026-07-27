# Raeen fuzz targets

Coverage-guided fuzzing (libFuzzer via [`cargo-fuzz`]) for the parsers that
consume **user-supplied, untrusted files** — an emulator's main input attack
surface. Targets must never panic, overflow an index, or OOM on any input;
they return errors instead.

| Target | Parser | Input it models |
|--------|--------|-----------------|
| `parse_elf` | `raeen_loader::elf::parse_elf` | eboot/module ELF bytes |
| `parse_pkg_header` | `raeen_loader::pkg::parse_pkg_header` | PKG file header |

## Running

Requires nightly and the `cargo-fuzz` tool (LLVM libFuzzer):

```bash
cargo install cargo-fuzz
cargo +nightly fuzz run parse_elf
cargo +nightly fuzz run parse_pkg_header
```

Crashing inputs are minimized into `fuzz/artifacts/<target>/`; add regressions
to the proptest suite in `crates/raeen-loader/tests/parser_robustness.rs` so
they stay fixed under plain `cargo test`.

This directory is excluded from the workspace (`[workspace] exclude`), so
normal builds and CI are unaffected.

[`cargo-fuzz`]: https://github.com/rust-fuzz/cargo-fuzz
