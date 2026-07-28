//! Multi-render-target acceptance: one draw with a fragment shader writing
//! Location 0 AND Location 1 renders into two colour attachments, and both
//! readbacks carry that draw's output — the second target is no longer
//! dropped. Also proves per-attachment state: the extra's write mask and
//! LOAD seed apply to it alone.
//!
//! Machines without Vulkan 1.3 skip (unless `RAEEN_REQUIRE_VULKAN=1`).

use kyty_graphics::spirv_asm::assemble;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::offscreen::{BlendState, DrawState, MrtAttachment, render_draw};
use raeen_gpu::vulkan::{VulkanBackend, shaders::triangle_vertex_spirv, validation_error_count};

const TOLERANCE: u8 = 2;

const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// Fragment shader with TWO colour outputs: green to Location 0, white to
/// Location 1 — the smallest shader that makes a dropped second attachment
/// observable.
const PS_DUAL_OUTPUT: &str = r#"
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outColor %outColor1
               OpExecutionMode %main OriginUpperLeft

               ; Annotations
               OpDecorate %outColor Location 0
               OpDecorate %outColor1 Location 1

               ; Types, variables and constants
       %void = OpTypeVoid
        %fty = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
  %outColor1 = OpVariable %_ptr_Output_v4float Output
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
      %green = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_1
      %white = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1

               ; Function main
       %main = OpFunction %void None %fty
        %lbl = OpLabel
               OpStore %outColor %green
               OpStore %outColor1 %white
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
    // Surface `[vulkan]` validation messages in `--nocapture` runs; the
    // count assertion alone cannot say WHICH rule fired.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("mrt_targets: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

#[test]
fn draw_writes_both_colour_attachments() {
    let ps = assemble(PS_DUAL_OUTPUT).expect("test PS must assemble");
    rspirv::dr::load_words(&ps).expect("test PS must parse as SPIR-V");

    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const W: u32 = 64;
    const H: u32 = 64;
    const MRT1_BASE: u64 = 0xBEEF_0000;

    let vs = triangle_vertex_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        mrt: vec![MrtAttachment {
            slot: 1,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            write_mask: ash::vk::ColorComponentFlags::RGBA,
            blend: BlendState::default(),
            target_base: MRT1_BASE,
            initial: None,
        }],
        ..DrawState::new(W, H, &vs, &ps)
    };

    let output = render_draw(dev, &state).expect("MRT draw must render");
    let primary = output.color.expect("colour draw produces a primary image");
    let center = primary.pixel(W / 2, H / 2).expect("in bounds");
    assert_pixel_eq(center, [0, 255, 0, 255], "primary center is green");

    assert_eq!(output.mrt_colors.len(), 1, "one extra attachment read back");
    let (base, extra) = &output.mrt_colors[0];
    assert_eq!(
        *base, MRT1_BASE,
        "extra readback is filed by its guest base"
    );
    assert_eq!((extra.width, extra.height), (W, H));
    let center = extra.pixel(W / 2, H / 2).expect("in bounds");
    assert_pixel_eq(center, [255, 255, 255, 255], "MRT1 center is white");
    // The extra had no seed: CLEAR to transparent black outside the triangle.
    let corner = extra.pixel(0, 0).expect("in bounds");
    assert_pixel_eq(corner, [0, 0, 0, 0], "MRT1 corner is the cleared colour");

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the MRT draw"
    );
}

/// Per-attachment state: the extra's write mask (R+A only) masks the white
/// output down on MRT1 while the primary still writes full RGBA; the extra's
/// LOAD seed survives outside the triangle.
#[test]
fn extra_attachment_write_mask_and_seed_apply_to_it_alone() {
    let ps = assemble(PS_DUAL_OUTPUT).expect("test PS must assemble");
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const W: u32 = 64;
    const H: u32 = 64;
    const MRT1_BASE: u64 = 0xBEEF_1000;

    // Seed MRT1 with opaque blue; the covered center overwrites only R and A.
    let mut seed = vec![0u8; (W * H * 4) as usize];
    for px in seed.chunks_exact_mut(4) {
        px.copy_from_slice(&[0, 0, 255, 255]);
    }

    let vs = triangle_vertex_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        mrt: vec![MrtAttachment {
            slot: 1,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            write_mask: ash::vk::ColorComponentFlags::R | ash::vk::ColorComponentFlags::A,
            blend: BlendState::default(),
            target_base: MRT1_BASE,
            initial: Some(seed),
        }],
        ..DrawState::new(W, H, &vs, &ps)
    };

    let output = render_draw(dev, &state).expect("seeded MRT draw must render");
    let (_, extra) = &output.mrt_colors[0];
    // Covered pixel: R and A written (white = 255), G untouched (0 from the
    // seed), B untouched (255 from the seed).
    let center = extra.pixel(W / 2, H / 2).expect("in bounds");
    assert_pixel_eq(center, [255, 0, 255, 255], "write mask masks G and B");
    // Uncovered pixel: the LOADed seed, byte-exact.
    let corner = extra.pixel(0, 0).expect("in bounds");
    assert_pixel_eq(corner, [0, 0, 255, 255], "seed survives outside coverage");
    // The primary is unaffected by the extra's mask.
    let primary = output.color.expect("primary image");
    let center = primary.pixel(W / 2, H / 2).expect("in bounds");
    assert_pixel_eq(center, [0, 255, 0, 255], "primary keeps full RGBA");

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the seeded MRT draw"
    );
}
