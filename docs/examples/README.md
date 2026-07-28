# Present plugin template

[`present-plugin-example.rs`](present-plugin-example.rs) is the supported
starting template for an out-of-tree Raeen present plugin. It is a complete,
dependency-free nearest-neighbour upscaler, not pseudocode. Raeen's integration
tests compile and load this exact file through the production plugin scanner.

## Try it unchanged

From the Raeen repository root:

```text
rustc --edition 2024 --crate-type cdylib --crate-name raeen_example_plugin -O --out-dir plugins docs/examples/present-plugin-example.rs
```

Start Raeen, open **Settings > Plugins**, and activate `example-nearest`.
Use **Rescan Plugins Folder** after rebuilding, or restart Raeen.

## Make your own

1. Copy `present-plugin-example.rs` into a separate repository.
2. Change the name returned by the `name` callback. Plugin names are selection
   keys, so keep yours stable and unique.
3. Replace the nearest-neighbour work in `process`, preserving all dimension,
   overflow, and buffer-length checks.
4. Set only the capability bits the implementation actually supports.
5. Keep output allocation paired with `release_output`; Raeen never frees a
   plugin allocation.
6. Build a `cdylib` and distribute the resulting platform shared library with
   its license and dependency instructions.

For a Cargo project, use:

```toml
[package]
name = "my-raeen-plugin"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]
```

Then copy the example to `src/lib.rs` and run:

```text
cargo build --release
```

Copy the resulting `.dll`, `.so`, or `.dylib` from `target/release/` into
Raeen's `plugins/` directory.

## ABI choice

- ABI v1 is sufficient for CPU-pixel spatial filters and upscalers.
- ABI v2 is the forward-compatible interface for GPU-frame plugins.
- Raeen currently supplies CPU frames only. A v2 plugin must decline unsupported
  frame kinds cleanly.

The complete ABI definitions, validation rules, current limitations, and
proprietary-code boundary are documented in
[`plugins/README.md`](../../plugins/README.md). Before publishing a binary,
also test malformed input, overflow, declined frames, repeated frames, and
output cleanup.
