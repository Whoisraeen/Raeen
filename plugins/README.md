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

Raeen itself ships only **GPL-2.0-clean** code. The default build includes the
`passthrough` and `nearest` reference plugins. Builds made with
`--features upscale-plugins` also include the in-tree `raeen-upscale` backends
described below. None of them link a proprietary vendor SDK.

## Available backends

The plugin system is vendor-neutral; it is not limited to DLSS.

| Backend | Status | Notes |
|---|---|---|
| `passthrough` | Working, built in | Identity/reference implementation |
| `nearest` | Working, built in | Simple reference upscaler |
| `bilinear` | Working, optional | Spatial 2x2 linear upscale |
| `bicubic` | Working, optional | Spatial Catmull-Rom upscale |
| `sharpen` | Working, optional | Native-resolution unsharp pass |
| `fsr` | Working, optional | Original FSR1-class EASU + RCAS implementation |
| `dlss` | Scaffolding/fallback only | Real inference needs a user-supplied NVIDIA runtime, GPU frames, depth, and motion vectors |
| `xess` | Scaffolding/fallback only | Real inference needs a user-supplied Intel runtime, GPU frames, depth, and motion vectors |

The optional backends are registered by `crates/raeen-upscale` when Raeen is
built with:

```text
cargo build -p raeen-gui --features upscale-plugins
```

FSR2/FSR3, DLSS, and XeSS are temporal techniques. The ABI reserves their
inputs, but Raeen does not yet populate GPU frames, depth, or motion vectors.
Until that work lands, selecting the current `dlss` or `xess` probe produces a
bicubic fallback rather than running the vendor technology.

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

## Two ways to write a plugin

| | In-tree Rust trait | **Out-of-tree C ABI** |
|---|---|---|
| Shape | crate compiled into `raeen-gpu` | standalone `.dll`/`.so`/`.dylib` |
| Loading | `register_present_plugin(...)` | dropped in `plugins/`, `dlopen`ed at startup |
| Rebuild Raeen? | yes | **no** |
| Language | Rust | anything with C linkage |
| For proprietary code | **no** — this links it | **yes** — nothing is linked |

**Use the C ABI for anything you cannot license under GPL-2.0.** It is the only
shape where the plugin is never linked into the Raeen artifact.

## The C ABI

Raeen scans `plugins/` at startup for files with the platform's shared-library
extension, loads each, and looks up one exported symbol:

```c
const RaeenPluginV1 *raeen_plugin_v1(void);
```

Return a pointer to a statically-lived struct whose `abi_version` is `1`.
Raeen then calls `create()` once, `name()` and `capabilities()` to describe the
plugin, `process()` per presented frame, `release_output()` after copying each
result, and `destroy()` at teardown.

```c
#include <stddef.h>
#include <stdint.h>

#define RAEEN_PLUGIN_ABI_VERSION 1u

#define RAEEN_CAP_UPSCALE              (1u << 0)
#define RAEEN_CAP_FRAME_GEN            (1u << 1)
#define RAEEN_CAP_WANTS_DEPTH          (1u << 2)
#define RAEEN_CAP_WANTS_MOTION_VECTORS (1u << 3)

#define RAEEN_OK 0

typedef struct { uint32_t width, height, bytes_per_texel, _reserved;
                 const uint8_t *data; size_t len; } RaeenAuxPlane;

typedef struct { uint32_t width, height, bytes_per_pixel, _reserved;
                 const uint8_t *color; size_t color_len;
                 const RaeenAuxPlane *depth;    /* NULL today */
                 const RaeenAuxPlane *motion;   /* NULL today */
                 uint64_t frame_index; } RaeenPresentFrame;

typedef struct { float output_scale; uint32_t hdr; } RaeenPresentContext;

typedef struct { uint32_t width, height, bytes_per_pixel, _reserved;
                 const uint8_t *pixels; size_t pixels_len; } RaeenPluginFrame;

typedef struct { RaeenPluginFrame primary;
                 const RaeenPluginFrame *generated;  /* reserved */
                 size_t generated_count; } RaeenPluginOutput;

typedef struct {
    uint32_t abi_version, _reserved;
    void  *(*create)(void);
    void   (*destroy)(void *inst);
    size_t (*name)(void *inst, uint8_t *buf, size_t cap);
    uint32_t (*capabilities)(void *inst);
    int32_t (*process)(void *inst, const RaeenPresentFrame *frame,
                       const RaeenPresentContext *ctx, RaeenPluginOutput *out);
    void   (*release_output)(void *inst, RaeenPluginOutput *out);
} RaeenPluginV1;
```

### Rules Raeen enforces

Break any of these and the frame is refused (the source frame is presented
unchanged) with a warning naming your plugin — never a silent no-op:

- `name` writes **at most `cap` bytes** and returns the count. No NUL needed.
  Return `0` or more than `cap` and the plugin is refused at load.
- `process` returns `RAEEN_OK` on success. **Any other value means "declined"** —
  a normal, cheap outcome for a temporal plugin that is still warming up.
- `bytes_per_pixel` on output must be **4** (display formats) or **8** (HDR
  `R16G16B16A16`).
- `pixels_len` must equal `width * height * bytes_per_pixel` **exactly**.
- Output edges are capped at 16384 and the buffer at 1 GiB.
- You may change the output resolution — that is what an upscaler is for.

### Memory ownership

**You allocate the output; you free it.** Raeen copies the pixels it needs and
then calls `release_output`, always — including when it rejected your output as
malformed. Neither side ever frees the other's allocation, so the two may use
different allocators, CRTs, and languages.

> **What Raeen cannot check:** `pixels_len` and the dimensions are both *your*
> claims. Raeen verifies they agree with each other, but nothing can verify
> either against your allocation's real size. Report a length you did not
> allocate and Raeen will read past your buffer. This is intrinsic to a C plugin
> boundary and is why plugins are user-supplied and opt-in.

## Working example — start here

A **complete, compiling, tested** reference plugin ships at
[`docs/examples/present-plugin-example.rs`](../docs/examples/present-plugin-example.rs):
a nearest-neighbour upscaler that honours `output_scale`, declines at native
scale, and manages its output memory correctly. It is deliberately
dependency-free and single-file, so it builds with one command:

```bash
rustc --edition 2024 --crate-type cdylib --crate-name raeen_example_plugin -O --out-dir plugins docs/examples/present-plugin-example.rs
```

Restart Raeen and it appears in **Settings ▸ Plugins** as `example-nearest`.

That file is not a sketch: the integration test
`crates/raeen-gpu/tests/present_plugin_dylib.rs` compiles **that exact source**
into a real shared library and loads it through the same `scan_dir` the Shell
uses, asserting the upscale is a correct 2D nearest map, that a declined frame
comes back as the source, and that 250 consecutive frames neither leak nor
fault. Copy it as your starting point.

For a short copy/build/test walkthrough, see
[`docs/examples/README.md`](../docs/examples/README.md).

## Sketch of the same thing, inline

`Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib"]
```

`src/lib.rs` — a passthrough that proves the wiring:

```rust
use std::ffi::c_void;

#[repr(C)]
pub struct RaeenPluginFrame {
    width: u32, height: u32, bytes_per_pixel: u32, _reserved: u32,
    pixels: *const u8, pixels_len: usize,
}
// ... the remaining #[repr(C)] structs, mirrored from the header above ...

unsafe extern "C" fn create() -> *mut c_void { Box::into_raw(Box::new(())).cast() }
unsafe extern "C" fn destroy(i: *mut c_void) {
    if !i.is_null() { drop(unsafe { Box::from_raw(i.cast::<()>()) }); }
}

unsafe extern "C" fn name(_i: *mut c_void, buf: *mut u8, cap: usize) -> usize {
    let n = b"my-upscaler";
    if cap < n.len() { return usize::MAX; }
    unsafe { std::ptr::copy_nonoverlapping(n.as_ptr(), buf, n.len()) };
    n.len()
}

unsafe extern "C" fn capabilities(_i: *mut c_void) -> u32 { 1 /* UPSCALE */ }

unsafe extern "C" fn process(
    _i: *mut c_void, frame: *const RaeenPresentFrame,
    _ctx: *const RaeenPresentContext, out: *mut RaeenPluginOutput,
) -> i32 {
    let (frame, out) = unsafe { (&*frame, &mut *out) };
    let src = unsafe { std::slice::from_raw_parts(frame.color, frame.color_len) };
    let pixels = src.to_vec();                    // your real work goes here
    let len = pixels.len();
    out.primary = RaeenPluginFrame {
        width: frame.width, height: frame.height,
        bytes_per_pixel: frame.bytes_per_pixel, _reserved: 0,
        pixels: Box::into_raw(pixels.into_boxed_slice()).cast(), pixels_len: len,
    };
    out.generated = std::ptr::null();
    out.generated_count = 0;
    0 // RAEEN_OK
}

unsafe extern "C" fn release_output(_i: *mut c_void, out: *mut RaeenPluginOutput) {
    let out = unsafe { &mut *out };
    if out.primary.pixels.is_null() { return; }
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        out.primary.pixels.cast_mut(), out.primary.pixels_len)) });
    out.primary.pixels = std::ptr::null();
}

static VTABLE: RaeenPluginV1 = RaeenPluginV1 {
    abi_version: 1, _reserved: 0,
    create, destroy, name, capabilities, process, release_output,
};

#[no_mangle]
pub extern "C" fn raeen_plugin_v1() -> *const RaeenPluginV1 { &VTABLE }
```

## Installing

Build, then drop the artifact in this directory:

```
plugins/my-upscaler.dll     (Windows)
plugins/libmy-upscaler.so   (Linux)
```

Restart Raeen. The plugin appears in **Settings ▸ Plugins** by the name it
reported. Refusals are logged with the reason and the filename — check the log
if yours does not appear.

### Package files

Raeen currently loads platform shared libraries directly; it does **not**
execute or install ZIP files. The supported distribution artifact today is:

```text
raeen_example_plugin.dll       Windows
libraeen_example_plugin.so     Linux
libraeen_example_plugin.dylib  macOS
```

A future `.raeen-plugin` archive can wrap a manifest, binaries, licenses, and
checksums, but it must be validated and extracted before the existing loader
loads its platform library. Renaming a ZIP or placing one in `plugins/` will
not load it. Keep package installation separate from the stable C ABI.

## In-tree Rust plugins

A GPL-2.0-compatible plugin (an MIT FSR pass, a community experiment) can skip
the C ABI and implement the trait directly:

```rust
use raeen_gpu::{PresentPlugin, PresentFrame, PresentContext, PluginOutput, Capabilities};

impl PresentPlugin for MyUpscaler {
    fn name(&self) -> &str { "my-upscaler" }
    fn capabilities(&self) -> Capabilities {
        Capabilities { upscale: true, ..Default::default() }
    }
    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        PluginOutput::identity(frame) // placeholder
    }
}

raeen_gpu::AgcGpuSession::register_present_plugin(Box::new(MyUpscaler::new()));
```

## ABI v2 — GPU-capable frames (for real hardware upscalers)

Real hardware upscalers and frame generators run as GPU passes and want a
`VkImage`, not a CPU pixel buffer. **ABI v2** is the stable contract for that,
designed so a plugin written today keeps working unchanged when Raeen's
GPU-resident present path lands:

```c
const RaeenPluginV2 *raeen_plugin_v2(void);   // abi_version == 2
```

- A binary may export **both** `raeen_plugin_v1` and `raeen_plugin_v2`; when
  v2 is present it is authoritative and must be valid.
- `create` receives a `RaeenHostContextV2` (valid only during the call):
  `host_flags` says what this host delivers, and a `RaeenVulkanContext`
  (instance/physical device/device/queue as opaque `u64`s plus a
  `vkGetInstanceProcAddr`) — **all zero today**, because the host does not yet
  set `RAEEN_HOST_GPU_FRAMES`.
- Every `RaeenPresentFrameV2` carries a `kind`: `RAEEN_FRAME_KIND_CPU`
  (delivered today; `color`/`color_len` have exactly the v1 semantics) or
  `RAEEN_FRAME_KIND_VULKAN` (a live `RaeenVulkanImage`, delivered once the
  GPU-resident present path exists). A plugin that only supports one kind
  simply declines the other from `process` — a declined frame is presented
  unchanged, never an error.
- Output is the v1 CPU output plus a reserved `produced_image` +
  `produced_kind`; the host reads the GPU output only when it advertised
  `RAEEN_HOST_GPU_FRAMES` (never today — claiming a GPU output now is refused
  with a named warning).
- New capability bit: `RAEEN_CAP_GPU_FRAMES (1 << 4)` — "I can consume
  VULKAN-kind frames". Shown in Settings ▸ Plugins as `GPU`.

**The license boundary is identical to v1 and non-negotiable**: a proprietary
implementation (e.g. an NVIDIA DLSS or Intel XeSS shim) is authored and hosted
in a separate repository, obtained by the user, and dropped in here as a
binary. Raeen ships none of it, fetches none of it, and this ABI stays
vendor-neutral — an MIT-licensed FSR pass fills the same slot in-tree, which
is what makes this an extension point rather than a copyleft-evasion device.

## Current limits

- **CPU-kind frames only for now** — the host does not yet set
  `RAEEN_HOST_GPU_FRAMES`; v2 plugins receive CPU buffers through v2 types
  until the GPU-resident present path lands (see
  `docs/superpowers/plans/2026-07-27-gpu-resident-present.md`).
- **`depth` and `motion` are always NULL.** The fields and capability bits exist
  so an MV-aware plugin can be written against a stable ABI now; PM4-side
  extraction is the follow-up that populates them — and it is the structural
  advantage: an external overlay must guess motion, Raeen owns the command
  stream and can eventually hand plugins the title's real motion vectors.
- **`generated` frames are validated but not presented.** Frame-gen pacing is
  not implemented, so a frame generator can be developed and its output checked,
  but the extra frames are not yet scheduled for display.
