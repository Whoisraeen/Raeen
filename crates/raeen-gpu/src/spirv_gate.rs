//! Translate-time SPIR-V validity gate.
//!
//! `vkCreateShaderModule` returns `VK_SUCCESS` for structurally invalid
//! SPIR-V — validation layers log the error but do not fail the call — and
//! dispatching such a module is undefined behavior: measured as an AMD driver
//! access violation that kills the whole process (ASTRO.BOT, 2026-07-21).
//! This gate runs the real Khronos validator (spirv-val, compiled in via the
//! `spirv-tools` crate) over every freshly translated module so an invalid
//! one becomes a named, negatively-cached translate failure that draws and
//! dispatches skip — the driver never sees it.
//!
//! naga deliberately does not serve here: its SPIR-V front end is itself a
//! structurizer and ACCEPTS back-edge modules spirv-val (and drivers) reject.

use spirv_tools::val::{self, Validator};

/// Validate `words` exactly the way the Vulkan validation layer does
/// (`spirv-val --relax-block-layout --target-env vulkan1.3`). `Err` carries a
/// one-line reason suitable for a named translate failure.
pub fn validate_spirv(words: &[u32]) -> Result<(), String> {
    let validator = val::create(Some(spirv_tools::TargetEnv::Vulkan_1_3));
    let options = val::ValidatorOptions {
        relax_block_layout: Some(true),
        ..Default::default()
    };
    validator.validate(words, Some(options)).map_err(|e| {
        // The diagnostic's first line is the actionable message; drop the
        // multi-line disassembly context spirv-val appends.
        let msg = e.to_string();
        let first = msg.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        format!("spirv-val: {first}")
    })
}

/// Whether the gate is enabled (default on; `RAEEN_SPIRV_GATE=0` bypasses it
/// for debugging — with the gate off, invalid modules reach the driver and
/// may kill the process).
pub fn gate_enabled() -> bool {
    std::env::var_os("RAEEN_SPIRV_GATE").is_none_or(|v| v != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid compute module: one entry point, one empty function.
    fn minimal_valid_module() -> Vec<u32> {
        vec![
            0x0723_0203, // magic
            0x0001_0300, // version 1.3
            0,           // generator
            10,          // bound
            0,           // schema
            (2 << 16) | 17,
            1, // OpCapability Shader
            (3 << 16) | 14,
            0,
            1, // OpMemoryModel Logical GLSL450
            (5 << 16) | 15,
            5,
            1,
            0x6e69_616d,
            0, // OpEntryPoint GLCompute %1 "main"
            (6 << 16) | 16,
            1,
            17,
            1,
            1,
            1, // OpExecutionMode %1 LocalSize 1 1 1
            (2 << 16) | 19,
            2, // %2 = OpTypeVoid
            (3 << 16) | 33,
            3,
            2, // %3 = OpTypeFunction %2
            (5 << 16) | 54,
            2,
            1,
            0,
            3, // %1 = OpFunction %2 None %3
            (2 << 16) | 248,
            4, // %4 = OpLabel
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56,  // OpFunctionEnd
        ]
    }

    #[test]
    fn accepts_a_minimal_valid_module() {
        assert_eq!(validate_spirv(&minimal_valid_module()), Ok(()));
    }

    #[test]
    fn rejects_a_back_edge_without_a_loop_header() {
        // The guest translator's measured failure class: a branch back to an
        // already-seen block with no OpLoopMerge ("Back-edges can only be
        // formed between a block and a loop header").
        let mut words = minimal_valid_module();
        let ret_at = words.len() - 2;
        assert_eq!(words[ret_at], (1 << 16) | 253);
        words[ret_at] = (2 << 16) | 249; // OpBranch
        words.insert(ret_at + 1, 4); // target = own block %4
        let err = validate_spirv(&words).expect_err("back-edge must be refused");
        assert!(err.starts_with("spirv-val:"), "named reason, got: {err}");
    }

    #[test]
    fn rejects_an_unstructured_selection_exit() {
        // The other measured class: a block inside a selection construct
        // jumping past the declared merge block ("exits the selection ...
        // not via a structured exit").
        let mut words = minimal_valid_module();
        // Truncate after "%4 = OpLabel" (drop OpReturn + OpFunctionEnd)...
        words.truncate(words.len() - 2);
        // ...but the types must come before the function: rebuild the tail.
        // Insert %5 = OpTypeBool, %6 = OpConstantTrue before OpFunction.
        let func_at = words
            .windows(2)
            .position(|w| w[0] == (5 << 16) | 54)
            .expect("OpFunction present");
        words.splice(
            func_at..func_at,
            [
                (2 << 16) | 20,
                5, // %5 = OpTypeBool
                (3 << 16) | 41,
                5,
                6, // %6 = OpConstantTrue %5
            ],
        );
        words.extend_from_slice(&[
            (3 << 16) | 247,
            8,
            0, // OpSelectionMerge %8 None
            (4 << 16) | 250,
            6,
            7,
            8, // OpBranchConditional %6 %7 %8
            (2 << 16) | 248,
            7, // %7 = OpLabel   (inside the selection)
            (2 << 16) | 249,
            9, // OpBranch %9   (bypasses merge %8 — invalid)
            (2 << 16) | 248,
            8, // %8 = OpLabel   (merge)
            (2 << 16) | 249,
            9, // OpBranch %9
            (2 << 16) | 248,
            9, // %9 = OpLabel
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56,  // OpFunctionEnd
        ]);
        assert!(
            validate_spirv(&words).is_err(),
            "selection bypass must be refused"
        );
    }

    #[test]
    fn rejects_garbage_words() {
        assert!(validate_spirv(&[0xdead_beef, 1, 2, 3]).is_err());
    }
}
