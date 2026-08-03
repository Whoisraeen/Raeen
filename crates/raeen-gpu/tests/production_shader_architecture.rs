//! Architecture gate for the commercial shader-translation path.
//!
//! Raeen previously carried a second, unused RDNA2 -> IR -> SPIR-V translator
//! under `src/shader/`. Unknown instructions in that prototype could lower to
//! no-ops, and comments incorrectly described it as the production path. Keep
//! the commercial path single-owner so future shader work cannot land in a
//! dead translator by mistake.

use std::fs;
use std::path::PathBuf;

#[test]
fn commercial_shader_translation_has_one_source_owner() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let retired_prototype = crate_root.join("src").join("shader");
    assert!(
        !retired_prototype.exists(),
        "the retired non-production shader translator returned at {}; commercial shaders must remain owned by kyty-graphics through shader_fetch",
        retired_prototype.display()
    );

    let lib = fs::read_to_string(crate_root.join("src").join("lib.rs"))
        .expect("raeen-gpu src/lib.rs must be readable");
    assert!(
        !lib.lines().any(|line| line.trim() == "pub mod shader;"),
        "src/lib.rs must not export the retired prototype translator"
    );

    let production = fs::read_to_string(crate_root.join("src").join("shader_fetch.rs"))
        .expect("production shader_fetch.rs must be readable");
    assert!(
        production.contains("kyty_graphics::shader::recompile"),
        "commercial shader translation must remain explicitly routed through kyty-graphics"
    );
}
