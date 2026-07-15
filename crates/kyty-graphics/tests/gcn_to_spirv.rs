//! Acceptance: the full Kyty GCN -> SPIR-V chain, driven through the crate's
//! public API only, and gated on `naga` parse + validate.
//!
//! This composes the stages exactly as Kyty's `GraphicsRender.cpp` does at
//! `CreatePipeline` (L2636-2639) -- which is the *only* place Kyty joins them,
//! since Kyty has no single combined entry point:
//!
//! ```text
//! auto vs_code   = ShaderParseVS(&vs_regs, &sh_regs);          // Shader.cpp L2287
//! ShaderGetInputInfoVS(&vs_regs, &sh_regs, &vs_input_info);    // Shader.cpp L1630
//! auto vs_shader = ShaderRecompileVS(vs_code, vs_input_info);  // Shader.cpp L2361
//!                    -> SpirvRun -> Assemble                   // Shader.cpp L845
//! ```
//!
//! `GraphicsRender.cpp` is not yet ported, so until it lands this test *is*
//! the composition site. Unlike the unit tests in `recompile.rs` -- which call
//! `shader_parse` directly on a bare instruction body and hand-build the
//! `input_info` -- this drives the real **analysis** stage: the shader blob
//! carries a genuine `0xBEEB03FF` binary-info trailer, and the input infos are
//! *derived* from hardware registers by `shader_get_input_info_{vs,ps}` rather
//! than being filled in by hand.

use kyty_graphics::shader::analysis::SHADER_BINARY_INFO_SENTINEL;
use kyty_graphics::shader::hw_regs::{PsStageRegisters, VsStageRegisters};
use kyty_graphics::shader::{
    PixelShaderInfo, ShaderMap, ShaderMemory, ShaderPixelInputInfo, ShaderRegisters,
    ShaderVertexInputInfo, VertexShaderInfo, shader_get_input_info_ps, shader_get_input_info_vs,
    shader_parse_ps, shader_parse_vs, shader_recompile_ps, shader_recompile_vs,
};

/// Kyty: `s_endpgm`.
const S_ENDPGM: u32 = 0xBF81_0000;

/// Guest memory backed by (base address, dwords) regions.
struct TestMem {
    regions: Vec<(u64, Vec<u32>)>,
}

impl ShaderMemory for TestMem {
    fn dwords_at(&self, addr: u64) -> Option<&[u32]> {
        if addr == 0 {
            return None;
        }
        for (base, data) in &self.regions {
            let end = base + data.len() as u64 * 4;
            if addr >= *base && addr < end && (addr - base) % 4 == 0 {
                return Some(&data[((addr - base) / 4) as usize..]);
            }
        }
        None
    }
}

/// Build a shader blob with the real Kyty trailer layout (`GetBinaryInfo`
/// Shader.cpp L909 / `GetUsageSlots` L921).
///
/// The blob opens with the `0xBEEB03FF` sentinel, which is itself a valid
/// SOP1 `s_mov_b32 s107, <literal>` whose literal operand doubles as the
/// trailer locator -- the binary info sits at `(code[1] + 1) * 2` dwords. The
/// instruction stream therefore walks the sentinel naturally before reaching
/// `body`.
///
/// Layout: `sentinel, offset_literal, body[, pad], slots[], usage_masks,
/// binary_info[7]`.
fn build_shader_blob(body: &[u32], slots: &[[u8; 4]], hash0: u32, crc32: u32) -> Vec<u32> {
    let mut v = vec![SHADER_BINARY_INFO_SENTINEL, 0];
    v.extend_from_slice(body);
    // The trailer must start at an even dword: (code[1] + 1) * 2.
    if (v.len() + slots.len() + 1) % 2 != 0 {
        v.push(0);
    }
    for s in slots {
        v.push(
            u32::from(s[0])
                | (u32::from(s[1]) << 8)
                | (u32::from(s[2]) << 16)
                | (u32::from(s[3]) << 24),
        );
    }
    v.push(0); // usage masks
    let info_dw = v.len();
    v[1] = (info_dw / 2 - 1) as u32;
    v.push(u32::from_le_bytes(*b"OrbS"));
    v.push(u32::from_le_bytes([b'h', b'd', b'r', 0x42])); // version 0x42
    v.push((body.len() as u32 * 4) << 8); // length
    v.push(1 | ((slots.len() as u32) << 8)); // chunk_usage_base_offset_dw = 1, num_slots
    v.push(hash0);
    v.push(0x1111_2222); // hash1
    v.push(crc32);
    v
}

/// Parse + validate with naga, returning the module so callers can assert on
/// its shape. A module that parses but fails validation is not a pass --
/// validation is the honest gate.
fn naga_parse_and_validate(words: &[u32], name: &str) -> naga::Module {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default())
        .unwrap_or_else(|e| panic!("naga parse of {name} failed: {e:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("naga validate of {name} failed: {e:?}"));
    module
}

/// Assert the validated module is a real single-entry shader of `stage` --
/// guards against a module that validates only because it is trivial.
fn assert_entry_point(module: &naga::Module, stage: naga::ShaderStage, name: &str) {
    assert_eq!(
        module.entry_points.len(),
        1,
        "{name}: expected exactly one entry point"
    );
    assert_eq!(module.entry_points[0].stage, stage, "{name}: wrong stage");
    assert_eq!(module.entry_points[0].name, "main", "{name}: entry name");
}

const VS_ADDR: u64 = 0x1_0000;
const PS_ADDR: u64 = 0x2_0000;

/// Real GCN vertex-shader body (same encodings the `recompile.rs` unit tests
/// cover, here wrapped in a genuine binary-info blob):
///
/// ```text
/// v_mov_b32 v0, lit(1.0f)      7E0002FF 3F800000
/// v_mov_b32 v1, 0              7E020280
/// v_mul_f32 v2, v0, v1         10040300
/// exp pos0   v0..v3 done       F80008CF 03020100
/// exp param0 v0..v3            F800020F 03020100
/// s_endpgm                     BF810000
/// ```
const VS_BODY: &[u32] = &[
    0x7E00_02FF,
    0x3F80_0000,
    0x7E02_0280,
    0x1004_0300,
    0xF800_08CF,
    0x0302_0100,
    0xF800_020F,
    0x0302_0100,
    S_ENDPGM,
];

/// Real GCN pixel-shader body:
///
/// ```text
/// v_interp_p1_f32 v2, v0, attr0.x   C8080000
/// v_interp_p2_f32 v2, v1, attr0.x   C8090001
/// v_mul_f32 v0, v2, v2              10000502
/// exp mrt0 v0, v0 compr vm done     F8001C0F 00000000
/// s_endpgm                          BF810000
/// ```
const PS_BODY: &[u32] = &[
    0xC808_0000,
    0xC809_0001,
    0x1000_0502,
    0xF800_1C0F,
    0x0000_0000,
    S_ENDPGM,
];

/// `user_sgpr` / `vs_user_sgpr.count` are left at their `0` defaults, which
/// satisfies the `user_sgpr <= count` guard in `shader_parse_vs`.
fn vs_regs() -> VertexShaderInfo {
    VertexShaderInfo {
        vs_regs: VsStageRegisters {
            data_addr: VS_ADDR,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ps_regs() -> PixelShaderInfo {
    PixelShaderInfo {
        ps_regs: PsStageRegisters {
            data_addr: PS_ADDR,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sh_regs() -> ShaderRegisters {
    let mut sh = ShaderRegisters {
        // GetExportCount() = 1 + ((spi_vs_out_config >> 1) & 0x1F); the 0
        // default therefore already yields the single export we want.
        //
        // Kyty (Shader.cpp L1759): ps_info->input_num = sh->ps_in_control.
        // PS_BODY interpolates attr0, so the hardware must advertise one
        // interpolant -- this is what makes analysis declare the %attr0 input
        // variable. Leaving it 0 makes recompile fail with
        // "id %attr0 is used but never defined".
        ps_in_control: 1,
        ..Default::default()
    };
    // Kyty target_output_mode 4 selects the FP16 (compr) MRT0 write that
    // PS_BODY's `exp mrt0 ... compr` performs.
    sh.target_output_mode[0] = 4;
    sh
}

fn mem() -> TestMem {
    TestMem {
        regions: vec![
            (
                VS_ADDR,
                build_shader_blob(VS_BODY, &[], 0xAAAA_0001, 0xBBBB_0001),
            ),
            (
                PS_ADDR,
                build_shader_blob(PS_BODY, &[], 0xAAAA_0002, 0xBBBB_0002),
            ),
        ],
    }
}

/// The full chain for a vertex shader:
/// GCN bytes -> parse -> **analysis** -> recompile -> assemble -> naga-valid.
#[test]
fn full_chain_vs_gcn_bytes_to_validated_spirv() {
    let mem = mem();
    let shader_map = ShaderMap::new();
    let regs = vs_regs();
    let sh = sh_regs();

    // 1. ShaderParseVS -- reads the blob out of guest memory, finds the
    //    0xBEEB03FF trailer, and takes hash0/crc32 from it.
    let code = shader_parse_vs(&regs, &sh, &mem, false).expect("shader_parse_vs");
    assert_eq!(code.get_hash0(), 0xAAAA_0001, "hash0 from binary info");
    assert_eq!(code.get_crc32(), 0xBBBB_0001, "crc32 from binary info");

    // 2. ShaderGetInputInfoVS -- input info *derived* from hardware registers.
    let mut input_info = ShaderVertexInputInfo::default();
    shader_get_input_info_vs(&regs, &sh, &mem, &shader_map, false, &mut input_info)
        .expect("shader_get_input_info_vs");
    assert_eq!(
        input_info.export_count, 1,
        "export count derived from spi_vs_out_config"
    );

    // 3. ShaderRecompileVS -> SpirvRun -> spirv_asm::assemble.
    let words = shader_recompile_vs(&code, &input_info).expect("shader_recompile_vs");

    // SPIR-V magic + non-empty module.
    assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    assert!(words.len() > 5, "module must have a body");

    // 4. The honest gate.
    let module = naga_parse_and_validate(&words, "full-chain vs");
    assert_entry_point(&module, naga::ShaderStage::Vertex, "full-chain vs");
}

/// The same full chain for a pixel shader. PS input info additionally depends
/// on the VS info (Kyty threads `vs_info` into `ShaderGetInputInfoPS`).
#[test]
fn full_chain_ps_gcn_bytes_to_validated_spirv() {
    let mem = mem();
    let shader_map = ShaderMap::new();
    let sh = sh_regs();

    let v_regs = vs_regs();
    let mut vs_input_info = ShaderVertexInputInfo::default();
    shader_get_input_info_vs(&v_regs, &sh, &mem, &shader_map, false, &mut vs_input_info)
        .expect("shader_get_input_info_vs");

    let regs = ps_regs();
    let code = shader_parse_ps(&regs, &sh, &mem, false).expect("shader_parse_ps");
    assert_eq!(code.get_hash0(), 0xAAAA_0002, "hash0 from binary info");

    let mut ps_input_info = ShaderPixelInputInfo::default();
    shader_get_input_info_ps(
        &regs,
        &sh,
        &vs_input_info,
        &mem,
        &shader_map,
        false,
        &mut ps_input_info,
    )
    .expect("shader_get_input_info_ps");

    let words = shader_recompile_ps(&code, &ps_input_info).expect("shader_recompile_ps");

    assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    let module = naga_parse_and_validate(&words, "full-chain ps");
    assert_entry_point(&module, naga::ShaderStage::Fragment, "full-chain ps");
}

/// A shader blob with no `0xBEEB03FF` trailer must fail in analysis with a
/// named error rather than producing a bogus module -- the boundary is
/// reported, not papered over.
#[test]
fn missing_binary_info_is_a_named_error() {
    let mem = TestMem {
        // Bare instruction body, no trailer.
        regions: vec![(VS_ADDR, VS_BODY.to_vec())],
    };
    let err = shader_parse_vs(&vs_regs(), &sh_regs(), &mem, false).unwrap_err();
    assert!(
        matches!(
            err,
            kyty_graphics::shader::ShaderAnalysisError::NoBinaryInfo
        ),
        "expected NoBinaryInfo, got {err:?}"
    );
}
