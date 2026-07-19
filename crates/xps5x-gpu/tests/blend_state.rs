//! Alpha-blend acceptance: a draw with `CB_BLEND0_CONTROL`-derived state
//! (SRC_ALPHA / ONE_MINUS_SRC_ALPHA / ADD) over a seeded opaque-red target
//! reads back the blended colour, not the source or destination alone.
//! Exercises the full path: `BlendState` on `DrawState` → pipeline blend
//! attachment → LOAD of the seeded attachment.
//!
//! Machines without Vulkan 1.3 skip (unless `XPS5X_REQUIRE_VULKAN=1`).

use kyty_graphics::spirv_asm::assemble;
use xps5x_gpu::backend::GpuBackend;
use xps5x_gpu::vulkan::offscreen::{BlendState, DrawState, render_draw};
use xps5x_gpu::vulkan::{VulkanBackend, shaders::triangle_vertex_spirv, validation_error_count};

const TOLERANCE: u8 = 2;

const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// Fragment shader: constant (0, 1, 0, 0.5) — green at half alpha.
const PS_HALF_GREEN: &str = r#"
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outColor
               OpExecutionMode %main OriginUpperLeft

               ; Annotations
               OpDecorate %outColor Location 0

               ; Types, variables and constants
       %void = OpTypeVoid
        %fty = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
  %float_half = OpConstant %float 0.5
      %color = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_half

               ; Function main
       %main = OpFunction %void None %fty
        %lbl = OpLabel
               OpStore %outColor %color
               OpReturn
               OpFunctionEnd
"#;

fn assert_pixel_eq(actual: [u8; 4], expected: [u8; 4], label: &str) {
    let close = actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE);
    assert!(
        close,
        "{label}: expected RGBA {expected:?} (+/-{TOLERANCE}), read back {actual:?}"
    );
}

fn backend_or_skip() -> Option<VulkanBackend> {
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("XPS5X_REQUIRE_VULKAN").is_none(),
                "XPS5X_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("blend_state: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

#[test]
fn alpha_blend_composites_over_the_seeded_target() {
    let ps = assemble(PS_HALF_GREEN).expect("test PS must assemble");
    rspirv::dr::load_words(&ps).expect("test PS must parse as SPIR-V");

    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const W: u32 = 64;
    const H: u32 = 64;

    // Seed the whole target with opaque red; the draw LOADs it (no clear).
    let mut initial = vec![0u8; (W * H * 4) as usize];
    for px in initial.chunks_exact_mut(4) {
        px.copy_from_slice(&[255, 0, 0, 255]);
    }

    let vs = triangle_vertex_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        initial: Some(&initial),
        blend: BlendState {
            enable: true,
            src_color: ash::vk::BlendFactor::SRC_ALPHA,
            dst_color: ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_op: ash::vk::BlendOp::ADD,
            src_alpha: ash::vk::BlendFactor::ONE,
            dst_alpha: ash::vk::BlendFactor::ZERO,
            alpha_op: ash::vk::BlendOp::ADD,
            constants: [0.0; 4],
        },
        ..DrawState::new(W, H, &vs, &ps)
    };

    let image = render_draw(dev, &state)
        .expect("blended draw must render")
        .color
        .expect("a colour draw produces a colour image");

    // 0.5 * green + 0.5 * red, alpha = 1 * 0.5 + 0 * 1 = 0.5.
    let center = image
        .pixel(W / 2, H / 2)
        .expect("center pixel is in bounds");
    assert_pixel_eq(
        center,
        [128, 128, 0, 128],
        "center should be the blended colour",
    );
    // Corners never saw the triangle: still the seeded red, proving the
    // initial contents were LOADed rather than cleared.
    let corner = image.pixel(0, 0).expect("corner pixel is in bounds");
    assert_pixel_eq(corner, [255, 0, 0, 255], "corner should stay seeded red");

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the blended draw"
    );
}
