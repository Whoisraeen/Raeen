//! End-to-end proof that a **real compiled plugin binary** loads and runs.
//!
//! The unit tests in `present_plugin::cabi` drive in-process vtables of
//! `extern "C"` functions. That exercises the whole adapter — validation,
//! ownership, teardown — but deliberately never touches `libloading`, so the
//! actual `LoadLibrary`/`dlopen` path had no positive coverage.
//!
//! This test closes that gap the honest way: it compiles the shipped reference
//! plugin (`docs/examples/present-plugin-example.rs`) into a real `cdylib` with
//! a bare `rustc` invocation, drops it in a temp directory, and loads it through
//! the same `scan_dir` the Shell uses at startup. The example users are told to
//! build is therefore the exact artifact this test verifies.
//!
//! `rustc` rather than `cargo` on purpose: a nested `cargo` would contend for
//! the workspace target-directory lock and can deadlock against the outer build.

use std::path::{Path, PathBuf};
use std::process::Command;

use raeen_gpu::present_plugin::cabi;
use raeen_gpu::present_plugin::{PresentContext, PresentFrame, PresentPlugin};

/// The reference plugin source, resolved from this crate's manifest directory
/// so the test does not depend on the working directory.
fn example_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/examples/present-plugin-example.rs")
        .canonicalize()
        .expect("the shipped reference plugin source must exist")
}

/// Directory the compiled plugin is written to: **under the Cargo target dir**,
/// beside this test binary — deliberately not `%TEMP%`.
///
/// Windows Application Control / Smart App Control scrutinises freshly-written
/// unsigned binaries in the user temp directory far more aggressively than build
/// output, and a blocked `LoadLibraryExW` made this suite flaky (see
/// [`policy_blocked`]). The target directory is also self-cleaning with
/// `cargo clean`.
fn plugin_out_dir() -> PathBuf {
    // current_exe() is <target>/<profile>/deps/<test>-<hash>.exe; go up to
    // <target>/<profile>/ so the artifact sits with the other build output.
    let exe = std::env::current_exe().expect("test executable path");
    let dir = exe
        .parent()
        .and_then(Path::parent)
        .expect("<target>/<profile>")
        .join("raeen-plugin-dylib-test");
    std::fs::create_dir_all(&dir).expect("plugin out dir");
    dir
}

/// Compile the reference plugin **once** for the whole suite and return its
/// directory.
///
/// Compiling once rather than per-test is both cheaper and materially less
/// flaky: each distinct freshly-written unsigned DLL is an independent chance
/// for Application Control to block the load, and three tests previously
/// produced three separate binaries.
///
/// Panics with rustc's own diagnostics if the example no longer compiles — a
/// broken reference example is a real defect, not a reason to skip.
fn example_plugin_dir() -> &'static Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = plugin_out_dir();
        let src = example_source();
        let out = Command::new("rustc")
            .args([
                "--edition",
                "2024",
                "--crate-type",
                "cdylib",
                "--crate-name",
                "raeen_example_plugin",
                "-C",
                "opt-level=1",
                "--out-dir",
            ])
            .arg(&dir)
            .arg(&src)
            .output()
            .expect("rustc must be available — cargo just used it to build this test");

        assert!(
            out.status.success(),
            "the shipped reference plugin failed to compile:\n--- stderr ---\n{}\n--- stdout ---\n{}",
            String::from_utf8_lossy(&out.stderr),
            String::from_utf8_lossy(&out.stdout),
        );
        dir
    })
    .as_path()
}

/// Whether a load failure is the host's executable-policy refusing an unsigned
/// binary, rather than anything wrong with Raeen or the plugin.
///
/// Windows Application Control / Smart App Control / WDAC returns
/// `ERROR_WLDP_...` (observed: OS code **4551**, "An Application Control policy
/// has blocked this file") from `LoadLibraryExW`. Whether it fires depends on
/// machine policy and on binary reputation, so it is nondeterministic across
/// runs on the *same* code — precisely the shape that makes a suite untrustworthy
/// if treated as a failure. Matched on the numeric code in the `Debug` rendering
/// so the check is locale-independent, with the message text as a fallback.
fn policy_blocked(err: &cabi::LoadError) -> bool {
    let detail = format!("{err:?}");
    detail.contains("code: 4551") || detail.contains("Application Control policy")
}

/// Load the compiled reference plugin, or `None` when host policy blocked it.
///
/// A `None` is reported loudly and the caller skips: this suite exists to prove
/// Raeen's loader works, and it cannot prove that on a machine that refuses to
/// load unsigned libraries at all.
fn load_example_plugin() -> Option<cabi::DynamicPlugin> {
    let dir = example_plugin_dir();
    // SAFETY: the directory contains only the plugin this suite compiled from
    // in-tree source — the user-controlled-input contract of `scan_dir`.
    let mut loaded = unsafe { cabi::scan_dir(dir) };
    assert_eq!(
        loaded.len(),
        1,
        "scan_dir must find exactly one compiled plugin in {}",
        dir.display()
    );
    match loaded.pop().unwrap() {
        Ok(plugin) => Some(plugin),
        Err(err) if policy_blocked(&err) => {
            eprintln!(
                "SKIP: host executable policy blocked the unsigned test plugin, so the \
                 real-dylib load path cannot be exercised here: {err}"
            );
            None
        }
        Err(err) => panic!("the reference plugin must load through the real dlopen path: {err}"),
    }
}

fn frame<'a>(w: u32, h: u32, buf: &'a [u8]) -> PresentFrame<'a> {
    PresentFrame {
        width: w,
        height: h,
        bytes_per_pixel: 4,
        color: buf,
        depth: None,
        motion: None,
        frame_index: 1,
    }
}

#[test]
fn a_real_compiled_plugin_binary_loads_and_upscales() {
    let dir = example_plugin_dir();

    // Prove the artifact is genuinely on disk with the platform extension
    // before claiming the loader found anything.
    let produced: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case(cabi::plugin_extension()))
        })
        .collect();
    assert_eq!(
        produced.len(),
        1,
        "expected exactly one {} artifact in {}, found {produced:?}",
        cabi::plugin_extension(),
        dir.display()
    );

    let Some(mut plugin) = load_example_plugin() else {
        return;
    };

    assert_eq!(
        plugin.name(),
        "example-nearest",
        "the name must survive the length-bounded name protocol"
    );
    assert!(
        plugin.capabilities().upscale,
        "the example advertises RAEEN_CAP_UPSCALE"
    );
    assert!(
        !plugin.capabilities().frame_gen,
        "unset capability bits must not appear"
    );

    // A 2x2 frame with four distinguishable pixels, upscaled 2x. Nearest
    // neighbour must replicate each source texel into a 2x2 block.
    let src: Vec<u8> = vec![
        0x10, 0x11, 0x12, 0x13, // (0,0)
        0x20, 0x21, 0x22, 0x23, // (1,0)
        0x30, 0x31, 0x32, 0x33, // (0,1)
        0x40, 0x41, 0x42, 0x43, // (1,1)
    ];
    let ctx = PresentContext {
        output_scale: 2.0,
        hdr: false,
    };
    let out = plugin.process(&frame(2, 2, &src), &ctx);

    assert_eq!((out.primary.width, out.primary.height), (4, 4));
    assert_eq!(out.primary.bytes_per_pixel, 4);
    assert_eq!(out.primary.pixels.len(), 4 * 4 * 4);
    assert!(
        out.generated.is_empty(),
        "a pure upscaler generates nothing"
    );

    let texel = |x: usize, y: usize| -> &[u8] {
        let i = (y * 4 + x) * 4;
        &out.primary.pixels[i..i + 4]
    };
    // Top-left 2x2 block is the source (0,0) texel, replicated.
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert_eq!(texel(x, y), &src[0..4], "block (0,0) at ({x},{y})");
    }
    // Bottom-right 2x2 block is the source (1,1) texel.
    for (x, y) in [(2, 2), (3, 2), (2, 3), (3, 3)] {
        assert_eq!(texel(x, y), &src[12..16], "block (1,1) at ({x},{y})");
    }
    // And the off-diagonal blocks are the other two source texels — proves the
    // sampling is a real 2D map, not a row or column smear.
    assert_eq!(texel(2, 0), &src[4..8], "block (1,0)");
    assert_eq!(texel(0, 2), &src[8..12], "block (0,1)");
}

#[test]
fn a_real_plugin_declining_at_native_scale_yields_the_source_frame() {
    let Some(mut plugin) = load_example_plugin() else {
        return;
    };

    // At scale 1.0 the example declines rather than allocating a byte-identical
    // duplicate; Raeen must then present the source unchanged.
    let src = vec![0x5Au8; 2 * 2 * 4];
    let ctx = PresentContext {
        output_scale: 1.0,
        hdr: false,
    };
    let out = plugin.process(&frame(2, 2, &src), &ctx);

    assert_eq!((out.primary.width, out.primary.height), (2, 2));
    assert_eq!(
        out.primary.pixels, src,
        "a declined frame must come back as the source, pixel-for-pixel"
    );
}

#[test]
fn a_real_plugin_survives_many_frames_without_leaking_or_faulting() {
    let Some(mut plugin) = load_example_plugin() else {
        return;
    };

    // The ownership contract (plugin allocates, `release_output` frees) is what
    // makes this safe to run per presented frame. A mistake there is a leak or
    // a double free per frame, so soak it rather than trusting one call.
    let src = vec![0x77u8; 8 * 8 * 4];
    let ctx = PresentContext {
        output_scale: 2.0,
        hdr: false,
    };
    for i in 0..250 {
        let mut f = frame(8, 8, &src);
        f.frame_index = i;
        let out = plugin.process(&f, &ctx);
        assert_eq!((out.primary.width, out.primary.height), (16, 16));
        assert_eq!(out.primary.pixels.len(), 16 * 16 * 4);
    }
}
