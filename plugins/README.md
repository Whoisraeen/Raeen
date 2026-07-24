# Out-of-tree present plugins (BYO)

This directory is where **user-supplied** present-path plugins — upscalers and
frame generators — live. Everything here **except this README is git-ignored**
(see the repo `.gitignore`). Raeen ships and distributes **nothing** from this
directory.

## Why this exists

Raeen has a generic, vendor-neutral plugin ABI in
[`crates/raeen-gpu/src/present_plugin`](../crates/raeen-gpu/src/present_plugin/mod.rs).
Any upscaler or frame generator — FSR, XeSS, a community experiment, or an
NVIDIA **DLSS** shim — implements the same `PresentPlugin` trait and registers
itself via `AgcGpuSession::register_present_plugin(...)`.

Raeen itself ships only the **built-in, GPL-2.0-clean** reference plugins
(`passthrough`, `nearest`, and — once integrated — an MIT-licensed FSR pass).
Those are all original Rust with no proprietary dependencies.

## The license boundary (read before adding a proprietary plugin)

Raeen is **GPL-2.0-only**. NVIDIA DLSS / Streamline binaries are proprietary and
are **not** GPL-2.0-compatible: they cannot be linked into, bundled with, or
fetched by a distributed Raeen build without making that build undistributable
under its own license. So the rule is strict and non-negotiable:

- Raeen's repository, releases, and installer contain **no** proprietary plugin
  code, and Raeen **never downloads** one on a user's behalf.
- A proprietary plugin (e.g. DLSS) is authored and hosted in a **separate**
  repository, obtained by the user themselves, and dropped in here. The
  combined DLSS-enabled build then exists only on that user's machine, assembled
  by the user for private use — which the GPL does not restrict.
- The ABI is deliberately **generic**. It is not a DLSS socket: FSR fills the
  same trait in-tree, which is what makes this a legitimate extension point
  rather than a copyleft-evasion device. Do not add DLSS-specific UI, branding,
  or "where to download it" instructions to the Raeen tree.

For the full reasoning see `docs/strategy/2026-07-23-go-to-ps5-emulator.md`
(§4.1) and the module docs in `present_plugin/mod.rs`.

## Writing a plugin (sketch)

A plugin is a crate that depends on `raeen-gpu` and implements the trait:

```rust
use raeen_gpu::{PresentPlugin, PresentFrame, PresentContext, PluginOutput, Capabilities};

pub struct MyUpscaler { /* ... */ }

impl PresentPlugin for MyUpscaler {
    fn name(&self) -> &str { "my-upscaler" }
    fn capabilities(&self) -> Capabilities {
        Capabilities { upscale: true, ..Default::default() }
    }
    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        // ... produce the upscaled / generated frame(s) ...
        PluginOutput::identity(frame) // placeholder
    }
}
```

Register and select it at startup:

```rust
raeen_gpu::AgcGpuSession::register_present_plugin(Box::new(MyUpscaler::new()));
raeen_gpu::AgcGpuSession::select_present_plugin("my-upscaler");
```

> Note: today the ABI is a compiled-in Rust trait (a plugin crate is built
> alongside Raeen by a user who opts in). A stable C-ABI `dlopen` layer for
> fully dynamic, no-recompile loading is a planned follow-up.
