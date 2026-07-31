//! Coverage bisect: reproduce the title path's draw shape IN-TREE, one
//! variable at a time, against the known-good fixture triangle shaders.
//!
//! Background (measured on retail Minecraft, 2026-07-20): every input to its
//! draws is verified correct — a textbook NDC quad, indices [0,1,2,0,2,3],
//! matching attribute bindings, gl_Position written, zero Vulkan validation
//! messages — yet ZERO fragments reach any render target (`RAEEN_FORCE_CLEAR`
//! proves the clear lands and nothing else does). Every previously PASSING
//! test drives the sink through the host `vertices` fixture path with the
//! default positive-height viewport; every TITLE draw uses the guest
//! `vertex_buffers` path with a Y-flipped viewport `[0, h, w, -h]` and often
//! an index buffer. Those three differences were never covered by a passing
//! test. Each test here flips exactly one of them, so a failure names the
//! guilty subsystem without a 170-second title run.
//!
//! Machines without Vulkan 1.3 skip (unless `RAEEN_REQUIRE_VULKAN=1`).

use ash::vk;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::offscreen::{
    CLEAR_COLOR, DrawState, IndexBinding, RenderedImage, ShaderStageBinding, StorageBufferBinding,
    TextureBinding, TextureUpload, VertexAttributeData, VertexBufferData, render_draw, unorm8,
};
use raeen_gpu::vulkan::shaders::{triangle_fragment_spirv, triangle_vertex_spirv};
use raeen_gpu::vulkan::{VulkanBackend, validation_error_count};

const W: u32 = 64;
const H: u32 = 64;

/// Same triangle as the passing fixture tests (`triangle_vertex_spirv` reads a
/// vec4 at Location 0, matching the fixture attribute R32G32B32A32_SFLOAT).
const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// Minecraft's measured quad: full-screen NDC corners at z = +1.0 (the far
/// plane, inclusive in Vulkan's 0..=w clip volume), w supplied as 1.0.
const QUAD_VERTICES: [[f32; 4]; 4] = [
    [-1.0, -1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, 1.0],
];

/// Minecraft's measured index stream: two triangles over the quad.
const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

fn backend_or_skip(name: &str) -> Option<VulkanBackend> {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("{name}: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

fn vec4s_to_bytes(vertices: &[[f32; 4]]) -> Vec<u8> {
    vertices
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// Pixels that differ from the clear colour — i.e. fragments some draw wrote.
fn non_clear_pixels(image: &RenderedImage) -> usize {
    assert_eq!(image.bytes_per_pixel, 4, "these tests render RGBA8");
    let clear = unorm8(CLEAR_COLOR);
    image
        .pixels
        .chunks_exact(4)
        .filter(|px| *px != clear)
        .count()
}

/// The guest `vertex_buffers` path (used by EVERY title draw, covered by NO
/// previously-passing test) with everything else at fixture defaults.
fn guest_triangle_state<'a>(vs: &'a [u32], ps: &'a [u32]) -> DrawState<'a> {
    DrawState {
        vertex_buffers: vec![VertexBufferData {
            bytes: vec4s_to_bytes(&TRIANGLE_VERTICES),
            stride: 16,
            per_instance: false,
        }],
        vertex_attributes: vec![VertexAttributeData {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 0,
        }],
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        ..DrawState::new(W, H, vs, ps)
    }
}

/// Variable 1: guest vertex-buffer upload alone (positive-height viewport).
#[test]
fn guest_vertex_buffer_path_renders() {
    let Some(backend) = backend_or_skip("guest_vertex_buffer_path_renders") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let (vs, ps) = (triangle_vertex_spirv(), triangle_fragment_spirv());

    let state = guest_triangle_state(&vs, &ps);
    let output = render_draw(dev, &state).expect("guest-path draw must render");
    let image = output.color.expect("colour draw produces an image");

    let covered = non_clear_pixels(&image);
    assert!(
        covered > 0,
        "the guest vertex-buffer path produced ZERO fragments for the same \
         triangle the host `vertices` path renders — the guest upload/binding \
         is the coverage bug"
    );
    assert_eq!(validation_error_count(), 0);
}

/// Variable 2: the title's Y-flipped viewport `[0, h, w, -h]` (guest path).
#[test]
fn y_flipped_viewport_still_renders() {
    let Some(backend) = backend_or_skip("y_flipped_viewport_still_renders") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let (vs, ps) = (triangle_vertex_spirv(), triangle_fragment_spirv());

    let state = DrawState {
        viewport: [0.0, H as f32, W as f32, -(H as f32)],
        ..guest_triangle_state(&vs, &ps)
    };
    let output = render_draw(dev, &state).expect("Y-flipped draw must render");
    let image = output.color.expect("colour draw produces an image");

    let covered = non_clear_pixels(&image);
    assert!(
        covered > 0,
        "a negative-height (Y-flip) viewport produced ZERO fragments — the \
         viewport handling is the coverage bug (every title draw uses this \
         shape; every previously-passing test used positive height)"
    );
    assert_eq!(validation_error_count(), 0);
}

/// Variable 4: the TITLE'S OWN translated vertex shader, replayed against the
/// measured quad with the known-good fixture fragment shader. Gated on
/// `RAEEN_TITLE_VS=<path to .spv>` (dump one with `RAEEN_DUMP_SHADERS`).
///
/// The sink is exonerated (variables 1–3 all cover), so the only untested
/// component from the title's zero-coverage draws is the translated shader
/// pair. The measured VS binds NO uniforms (pushc=0, sbuf=0, tex=0) so it can
/// run standalone. It reads a vec3 at Location 0 — the measured stride-12 NDC
/// quad. If THIS covers nothing while variables 1–3 cover, the translated VS
/// module (exec masking, entry structure — not the POS0 text, which the
/// passthrough probe already exercised) is the coverage bug.
#[test]
fn title_translated_vs_covers_the_measured_quad() {
    let Ok(vs_path) = std::env::var("RAEEN_TITLE_VS") else {
        eprintln!("title_translated_vs_covers_the_measured_quad: SKIP — set RAEEN_TITLE_VS");
        return;
    };
    let Some(backend) = backend_or_skip("title_translated_vs_covers_the_measured_quad") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let raw = std::fs::read(&vs_path).expect("RAEEN_TITLE_VS file must be readable");
    assert_eq!(raw.len() % 4, 0, "SPIR-V byte length must be word-aligned");
    let vs: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let ps = triangle_fragment_spirv();

    // Minecraft's measured quad exactly as the title supplies it: stride 12
    // (xyz only — the VS supplies w), vec3 attribute at Location 0.
    let quad3: [[f32; 3]; 4] = [
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let bytes: Vec<u8> = quad3
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let index_bytes: Vec<u8> = QUAD_INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();

    let state = DrawState {
        viewport: [0.0, H as f32, W as f32, -(H as f32)],
        vertex_buffers: vec![VertexBufferData {
            bytes,
            stride: 12,
            per_instance: false,
        }],
        vertex_attributes: vec![VertexAttributeData {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        }],
        vertex_count: QUAD_INDICES.len() as u32,
        index: Some(IndexBinding {
            bytes: &index_bytes,
            index_type: vk::IndexType::UINT16,
        }),
        ..DrawState::new(W, H, &vs, &ps)
    };
    let output = render_draw(dev, &state).expect("title-VS draw must render");
    let image = output.color.expect("colour draw produces an image");

    let covered = non_clear_pixels(&image);
    let total = (W * H) as usize;
    eprintln!("title VS replay: covered {covered}/{total} pixels");
    assert!(
        covered > 0,
        "the title's translated VS produced ZERO fragments on the exact quad \
         that covers {total}/{total} under the fixture VS — the translated VS \
         module is the coverage bug"
    );
}

/// Replay Minecraft's final composite vertex shader with the exact captured
/// two-attribute stream. This is intentionally gated on a local shader dump:
/// retail shader bytes are diagnostics and never repository fixtures.
#[test]
fn minecraft_composite_vs_covers_captured_quad() {
    let Ok(vs_path) = std::env::var("RAEEN_MINECRAFT_COMPOSITE_VS") else {
        eprintln!(
            "minecraft_composite_vs_covers_captured_quad: SKIP — set \
             RAEEN_MINECRAFT_COMPOSITE_VS"
        );
        return;
    };
    let Some(backend) = backend_or_skip("minecraft_composite_vs_covers_captured_quad") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let raw = std::fs::read(&vs_path).expect("composite VS must be readable");
    assert_eq!(raw.len() % 4, 0, "SPIR-V byte length must be word-aligned");
    let vs: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let ps = triangle_fragment_spirv();

    // Captured from Minecraft's 0x20040000 composite: position float4 and
    // texture-coordinate float3 in a 28-byte stride.
    let records: [[f32; 7]; 4] = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0],
    ];
    let bytes = records
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let index_bytes: Vec<u8> = QUAD_INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
    let state = DrawState {
        viewport: [0.0, H as f32, W as f32, -(H as f32)],
        vertex_buffers: vec![VertexBufferData {
            bytes,
            stride: 28,
            per_instance: false,
        }],
        vertex_attributes: vec![
            VertexAttributeData {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
            VertexAttributeData {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 16,
            },
        ],
        vertex_count: QUAD_INDICES.len() as u32,
        index: Some(IndexBinding {
            bytes: &index_bytes,
            index_type: vk::IndexType::UINT16,
        }),
        ..DrawState::new(W, H, &vs, &ps)
    };
    let output = render_draw(dev, &state).expect("captured composite VS draw must render");
    let image = output.color.expect("colour draw produces an image");
    let covered = non_clear_pixels(&image);
    assert!(
        covered > 0,
        "Minecraft's composite VS produced zero fragments with its exact \
         captured vertex stream"
    );
}

/// Variable 5: the TITLE'S translated pixel shaders, swept against the title
/// VS on the covering quad. Gated on `RAEEN_TITLE_SHADER_DIR` (a
/// `RAEEN_DUMP_SHADERS` directory). The VS is exonerated (variable 4 covers
/// 100%), so the last untested component is each PS — with plausible NON-ZERO
/// descriptor content, distinguishing "the PS translation kills every
/// fragment" (covers nothing here) from "the PS is fine but its title-runtime
/// inputs are zeroed" (covers here, black in the title).
///
/// Binding layout per `shader_calc_binding_indices`: storage buffers at
/// binding 0 (pushc 16/buffer); textures sampled=0/sampler=2 (pushc 48).
#[test]
fn title_translated_ps_sweep() {
    let Ok(dir) = std::env::var("RAEEN_TITLE_SHADER_DIR") else {
        eprintln!("title_translated_ps_sweep: SKIP — set RAEEN_TITLE_SHADER_DIR");
        return;
    };
    let vs_path = std::env::var("RAEEN_TITLE_VS").expect("also set RAEEN_TITLE_VS");
    let Some(backend) = backend_or_skip("title_translated_ps_sweep") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let read_spv = |p: &std::path::Path| -> Vec<u32> {
        std::fs::read(p)
            .expect("spv readable")
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let vs = read_spv(std::path::Path::new(&vs_path));

    let quad3: [[f32; 3]; 4] = [
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let quad_bytes: Vec<u8> = quad3
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let index_bytes: Vec<u8> = QUAD_INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();

    // Plausible non-zero resources: a storage buffer of 1.0f, a white 4x4
    // texture, a fabricated V# (stride 16, 4 records) in the push constants.
    // `RAEEN_TITLE_PS_ZERO=1` zeroes ALL of it instead — replicating what the
    // title runtime would feed if its fragment-stage resource capture is
    // broken. Non-zero covers + zero goes dark = the black frame reproduced
    // in-tree, pinning the bug on resource CONTENT, not translation.
    let zero = std::env::var_os("RAEEN_TITLE_PS_ZERO").is_some();
    let sbuf_binding = move || {
        Some(StorageBufferBinding {
            binding: 0,
            buffers: vec![std::sync::Arc::new(if zero {
                vec![0u8; 256]
            } else {
                (0..64)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect::<Vec<u8>>()
            })],
            guest_bases: vec![0],
            guest_sizes: vec![256],
            writable: vec![false],
        })
    };
    let tex_binding = move || {
        Some(TextureBinding {
            sampled_binding: 0,
            sampler_binding: 2,
            textures: vec![TextureUpload {
                width: 4,
                height: 4,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![if zero { 0x00 } else { 0xFF }; 4 * 4 * 4],
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: 0,
                sample_hash: 0,
                cached: false,
            }],
            samplers: vec![raeen_gpu::vulkan::offscreen::SamplerState::nearest_repeat()],
            sampled_groups: Vec::new(),
        })
    };
    let vsharp_pushc: Vec<u8> = if zero {
        vec![0u8; 16]
    } else {
        [0u32, 16 << 16, 4, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    };
    let tex_pushc: Vec<u8> = if zero {
        vec![0u8; 48]
    } else {
        (0..12u32).flat_map(|_| 4u32.to_le_bytes()).collect()
    };

    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("shader dir readable")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "spv")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("ps_"))
        })
        .collect();
    entries.sort();

    // Minimal SPIR-V scan: which resource classes does the module declare?
    // A mismatched pipeline layout is not a recoverable error on this driver
    // (measured: STATUS_ACCESS_VIOLATION inside vkCreateGraphicsPipelines),
    // so the config must be RIGHT the first time, not guessed by ladder.
    // OpVariable = 59: [type, id, storage-class]. Classes: UniformConstant=0
    // (sampled images/samplers), Uniform=2 / StorageBuffer=12 (storage
    // buffers), PushConstant=9.
    let scan = |words: &[u32]| -> (bool, bool, bool) {
        let (mut pushc, mut sbuf, mut tex) = (false, false, false);
        let mut i = 5;
        while i < words.len() {
            let op = words[i] & 0xffff;
            let len = (words[i] >> 16) as usize;
            if len == 0 {
                break;
            }
            if op == 59 && i + 3 < words.len() {
                match words[i + 3] {
                    9 => pushc = true,
                    2 | 12 => sbuf = true,
                    0 => tex = true,
                    _ => {}
                }
            }
            i += len;
        }
        (pushc, sbuf, tex)
    };

    for ps_path in entries {
        let ps = read_spv(&ps_path);
        let name = ps_path.file_name().unwrap().to_string_lossy().into_owned();

        // Build exactly the config the module declares. Binding numbers per
        // `shader_calc_binding_indices`: storage first, then sampled(+1 for
        // the storage-image slot), then sampler.
        let (pushc, sbuf, tex) = scan(&ps);
        let cfg_name = format!("scan pushc={pushc} sbuf={sbuf} tex={tex}");
        let (sampled_binding, sampler_binding) = if sbuf { (1, 3) } else { (0, 2) };
        let bindings = if sbuf || tex || pushc {
            vec![ShaderStageBinding {
                stage: vk::ShaderStageFlags::FRAGMENT,
                descriptor_set_slot: 0,
                push_constant_offset: 0,
                push_constants: if pushc {
                    if tex {
                        tex_pushc.clone()
                    } else {
                        vsharp_pushc.clone()
                    }
                } else {
                    Vec::new()
                },
                push_uniform_binding: None,
                storage_buffers: if sbuf { sbuf_binding() } else { None },
                textures: if tex {
                    let mut t = tex_binding().unwrap();
                    t.sampled_binding = sampled_binding;
                    t.sampler_binding = sampler_binding;
                    Some(t)
                } else {
                    None
                },
                storage_images: None,
                gds_binding: None,
                eud_raw: None,
                global_mem: None,
            }]
        } else {
            Vec::new()
        };

        let outcome;
        {
            let state = DrawState {
                viewport: [0.0, H as f32, W as f32, -(H as f32)],
                vertex_buffers: vec![VertexBufferData {
                    bytes: quad_bytes.clone(),
                    stride: 12,
                    per_instance: false,
                }],
                vertex_attributes: vec![VertexAttributeData {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                }],
                vertex_count: QUAD_INDICES.len() as u32,
                index: Some(IndexBinding {
                    bytes: &index_bytes,
                    index_type: vk::IndexType::UINT16,
                }),
                stage_bindings: bindings,
                ..DrawState::new(W, H, &vs, &ps)
            };
            match render_draw(dev, &state) {
                Ok(output) => {
                    let covered = output.color.map_or(0, |img| non_clear_pixels(&img));
                    outcome = format!("[{cfg_name}] covered {covered}/{}", (W * H) as usize);
                }
                Err(e) => {
                    let mut msg = e.to_string();
                    msg.truncate(100);
                    outcome = format!("[{cfg_name}] error: {msg}");
                }
            }
        }
        eprintln!("PS SWEEP {name}: {outcome}");
    }
}

/// Replay a captured title VS/PS pair with the descriptor ABI observed on the
/// failing ASTRO.BOT HDR composite, but with tiny deterministic resources.
/// This separates translated-shader execution from the title's 1080p target
/// and 2432x1368 source allocation. Gated because the SPIR-V dumps are local
/// diagnostics, never repository fixtures.
#[test]
fn captured_hdr_composite_shader_pair_submits_safely() {
    let (Ok(vs_path), Ok(ps_path)) = (
        std::env::var("RAEEN_REPLAY_VS"),
        std::env::var("RAEEN_REPLAY_PS"),
    ) else {
        eprintln!(
            "captured_hdr_composite_shader_pair_submits_safely: SKIP — set RAEEN_REPLAY_VS/PS"
        );
        return;
    };
    let Some(backend) = backend_or_skip("captured_hdr_composite_shader_pair_submits_safely") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let read_spv = |path: &str| {
        std::fs::read(path)
            .expect("captured SPIR-V readable")
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let (vs, ps) = (read_spv(&vs_path), read_spv(&ps_path));
    let (target_w, target_h) = if std::env::var_os("RAEEN_REPLAY_FULL_TARGET").is_some() {
        (1920, 1080)
    } else {
        (W, H)
    };
    let (texture_w, texture_h) = if std::env::var_os("RAEEN_REPLAY_FULL_TEXTURE").is_some() {
        (2432, 1368)
    } else {
        (4, 4)
    };

    // Full-screen triangle: position float4 + UV float2 in a 24-byte stride.
    // The second float4 descriptor overlaps the next record exactly like the
    // captured draw; only its first two lanes are consumed by the VS.
    let records: [[f32; 6]; 3] = [
        [-1.0, -1.0, 0.0, 1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0, 2.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0, 0.0, -1.0],
    ];
    let mut vertex_bytes = records
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect::<Vec<_>>();
    vertex_bytes.extend_from_slice(&[0; 8]); // final overlapped float4 tail
    if let Ok(path) = std::env::var("RAEEN_REPLAY_VERTEX") {
        vertex_bytes = std::fs::read(path).expect("captured vertex buffer readable");
    }

    let push_words: [u32; 16] = [
        0x0000_0000,
        0x0010_0000,
        0x0000_0002,
        0x0004_dfac,
        0x0000_0000,
        0xc470_0000,
        0x0155_c25f,
        0x91b0_0fac,
        0x0000_0000,
        0x0000_0000,
        0xe07b_0000,
        0x0005_7054,
        0x0000_0000,
        0x00ff_f000,
        0x0650_0000,
        0x0000_0000,
    ];
    let push_constants = push_words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();

    let storage_bytes = std::env::var("RAEEN_REPLAY_STORAGE")
        .ok()
        .map(|path| std::fs::read(path).expect("captured storage buffer readable"))
        .unwrap_or_else(|| vec![0; 32]);
    let texture_pixels = std::env::var("RAEEN_REPLAY_TEXTURE")
        .ok()
        .map(|path| std::fs::read(path).expect("captured decoded texture readable"))
        .unwrap_or_else(|| vec![0; (texture_w * texture_h * 8) as usize]);
    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::FRAGMENT,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants,
        push_uniform_binding: None,
        storage_buffers: Some(StorageBufferBinding {
            binding: 0,
            buffers: vec![std::sync::Arc::new(storage_bytes)],
            guest_bases: vec![0],
            guest_sizes: vec![256],
            writable: vec![false],
        }),
        textures: Some(TextureBinding {
            sampled_binding: 1,
            sampler_binding: 3,
            textures: vec![TextureUpload {
                width: texture_w,
                height: texture_h,
                format: vk::Format::R16G16B16A16_SFLOAT,
                pixels: texture_pixels,
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: 0,
                sample_hash: 0,
                cached: false,
            }],
            samplers: vec![raeen_gpu::vulkan::offscreen::SamplerState::nearest_repeat()],
            sampled_groups: Vec::new(),
        }),
        storage_images: None,
        gds_binding: None,
        eud_raw: None,
        global_mem: None,
    };

    let target_base = std::env::var_os("RAEEN_REPLAY_PERSISTENT")
        .is_some()
        .then_some(0x5_3aa0_0000);
    let state = DrawState {
        format: vk::Format::B10G11R11_UFLOAT_PACK32,
        topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
        target_base,
        vertex_buffers: vec![VertexBufferData {
            bytes: vertex_bytes,
            stride: 24,
            per_instance: false,
        }],
        vertex_attributes: vec![
            VertexAttributeData {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
            VertexAttributeData {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 16,
            },
        ],
        vertex_count: 3,
        stage_bindings: vec![binding],
        ..DrawState::new(target_w, target_h, &vs, &ps)
    };
    let output = render_draw(dev, &state).expect("captured HDR shader pair must submit safely");
    assert!(output.color.is_some(), "colour attachment must read back");
}

/// Variable 3: the full Minecraft shape — indexed NDC quad at z=+1.0, guest
/// vertex path, Y-flipped viewport. This is byte-for-byte the measured draw.
#[test]
fn indexed_fullscreen_quad_with_y_flip_covers_everything() {
    let Some(backend) = backend_or_skip("indexed_fullscreen_quad_with_y_flip_covers_everything")
    else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let (vs, ps) = (triangle_vertex_spirv(), triangle_fragment_spirv());

    let index_bytes: Vec<u8> = QUAD_INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
    let state = DrawState {
        viewport: [0.0, H as f32, W as f32, -(H as f32)],
        vertex_buffers: vec![VertexBufferData {
            bytes: vec4s_to_bytes(&QUAD_VERTICES),
            stride: 16,
            per_instance: false,
        }],
        vertex_attributes: vec![VertexAttributeData {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: 0,
        }],
        vertex_count: QUAD_INDICES.len() as u32,
        index: Some(IndexBinding {
            bytes: &index_bytes,
            index_type: vk::IndexType::UINT16,
        }),
        ..DrawState::new(W, H, &vs, &ps)
    };
    let output = render_draw(dev, &state).expect("indexed quad must render");
    let image = output.color.expect("colour draw produces an image");

    let covered = non_clear_pixels(&image);
    let total = (W * H) as usize;
    assert!(
        covered == total,
        "a full-screen NDC quad must cover every pixel; covered {covered}/{total} \
         — if 0, the indexed-draw path (or z=+1.0 far-plane clipping) is the \
         coverage bug"
    );
    assert_eq!(validation_error_count(), 0);
}
