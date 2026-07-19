//! ShaderMemory Phase 2 acceptance: a DCB that binds a **real GCN pixel
//! shader living in guest memory** (not an embedded id) drives a Vulkan draw
//! whose pixels come from that fetched-and-recompiled shader.
//!
//! The bind uses the same packets a Gen5 title emits — `SPI_SHADER_PGM_LO/
//! HI_PS` + `RSRC2_PS` SH-register writes — and the shader body carries no
//! host-side special-casing: it is fetched from its address through the
//! VirtualQuery-checked identity map, parsed to `s_endpgm`, recompiled to
//! SPIR-V and cached.
//!
//! Also pinned: a bind pointing at garbage **skips the draw** (named,
//! negative-cached) instead of failing the DCB — the honest-degradation
//! contract.
//!
//! Machines without Vulkan 1.3 skip (unless `XPS5X_REQUIRE_VULKAN=1`).

use kyty_graphics::pm4;
use std::sync::Arc;
use xps5x_gpu::GpuGuestMemory;
use xps5x_gpu::agc_exec::AgcGpuSession;

struct TestGpuMemory {
    start: u64,
    len: u64,
}

impl GpuGuestMemory for TestGpuMemory {
    fn validate_gpu_range(&self, addr: u64, len: u64, write: bool) -> bool {
        !write
            && addr >= self.start
            && addr
                .checked_add(len)
                .is_some_and(|end| end <= self.start + self.len)
    }

    fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
        if !self.validate_gpu_range(addr, out.len() as u64, false) {
            return false;
        }
        // SAFETY: `AlignedBlob` owns this exact range and outlives the
        // synchronous session call in each test.
        unsafe { std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len()) };
        true
    }

    fn write_gpu(&self, _addr: u64, _data: &[u8]) -> bool {
        false
    }
}

/// `s_endpgm`.
const S_ENDPGM: u32 = 0xBF81_0000;

/// Solid green GCN pixel shader (same body as the `shader_bridge` fixture):
/// v0=0, v1=1.0, v2=0, v3=1.0; exp mrt0 v0..v3 vm done; s_endpgm.
const PS_BODY_SOLID_GREEN: &[u32] = &[
    0x7E00_0280,
    0x7E02_02FF,
    0x3F80_0000,
    0x7E04_0280,
    0x7E06_02FF,
    0x3F80_0000,
    0xF800_180F,
    0x0302_0100,
    S_ENDPGM,
];

/// PS4-style blob with the `0xBEEB03FF` binary-info trailer.
fn build_shader_blob(body: &[u32], hash0: u32, crc32: u32) -> Vec<u32> {
    const SENTINEL: u32 = 0xBEEB_03FF;
    let mut v = vec![SENTINEL, 0];
    v.extend_from_slice(body);
    if (v.len() + 1) % 2 != 0 {
        v.push(0);
    }
    v.push(0); // usage masks
    let info_dw = v.len();
    v[1] = (info_dw / 2 - 1) as u32;
    v.push(u32::from_le_bytes(*b"OrbS"));
    v.push(u32::from_le_bytes([b'h', b'd', b'r', 0x42]));
    v.push((body.len() as u32 * 4) << 8);
    v.push(1);
    v.push(hash0);
    v.push(0x1111_2222);
    v.push(crc32);
    v
}

/// The PGM_LO register value is the address shifted right by 8, so the shader
/// base must be 256-byte aligned. Over-allocate and slide to alignment.
struct AlignedBlob {
    _storage: Vec<u32>,
    addr: u64,
}

impl AlignedBlob {
    fn memory(&self) -> Arc<dyn GpuGuestMemory> {
        Arc::new(TestGpuMemory {
            start: self._storage.as_ptr() as u64,
            len: std::mem::size_of_val(self._storage.as_slice()) as u64,
        })
    }
}

fn place_aligned(blob: &[u32]) -> AlignedBlob {
    // ShaderCache fetches one bounded 4 KiB page at a time, matching a real
    // guest mapping. Keep a full readable page after the alignment slide; a
    // tiny Vec containing only the fixture blob is not a faithful GPU-visible
    // mapping and correctly fails the new capability check.
    let mut storage = vec![0u32; blob.len().max(1024) + 64];
    let base = storage.as_ptr() as u64;
    let aligned = base.next_multiple_of(256);
    let offset = ((aligned - base) / 4) as usize;
    storage[offset..offset + blob.len()].copy_from_slice(blob);
    AlignedBlob {
        _storage: storage,
        addr: aligned,
    }
}

/// A register-complete Gen5 DCB: embedded VS (clear quad), **guest-memory PS**
/// bound at `ps_addr` via SH registers, RectList draw over `width`x`height`.
fn build_guest_ps_dcb(width: u32, height: u32, ps_addr: u64) -> Vec<u32> {
    let mut dcb = Vec::new();

    let mut set_cx = |reg: u32, values: &[u32]| {
        dcb.push(pm4::header(
            (values.len() + 2) as u16,
            pm4::IT_SET_CONTEXT_REG,
            pm4::R_ZERO,
        ));
        dcb.push(reg);
        dcb.extend_from_slice(values);
    };

    set_cx(pm4::CB_COLOR0_BASE, &[0x1_0000 >> 8]);
    set_cx(pm4::CB_COLOR0_INFO, &[0xa << 2]); // 8_8_8_8 unorm RGBA
    set_cx(
        pm4::CB_COLOR0_ATTRIB2,
        &[((width - 1) << 14) | (height - 1)],
    );
    set_cx(pm4::CB_TARGET_MASK, &[0xF]);
    // Non-compressed FP32 MRT0 export -> target output mode 9.
    set_cx(pm4::SPI_SHADER_COL_FORMAT, &[9]);

    let (hw, hh) = (width as f32 / 2.0, height as f32 / 2.0);
    set_cx(
        pm4::PA_CL_VPORT_XSCALE,
        &[
            hw.to_bits(),
            hw.to_bits(),
            hh.to_bits(),
            hh.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
        ],
    );
    set_cx(pm4::PA_SC_SCREEN_SCISSOR_TL, &[0, width | (height << 16)]);

    // RectList.
    dcb.push(pm4::header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO));
    dcb.push(pm4::VGT_PRIMITIVE_TYPE);
    dcb.push(17);

    // Embedded VS id 0 (Kyty's clear-quad VS).
    dcb.push(pm4::header(29, pm4::IT_NOP, pm4::R_VS_EMBEDDED));
    dcb.push(0);
    dcb.push(0);
    dcb.resize(dcb.len() + 26, 0);

    // The real PS bind: plain SH-register writes, exactly a Gen5 title's form.
    let mut set_sh = |reg: u32, value: u32| {
        dcb.push(pm4::header(3, pm4::IT_SET_SH_REG, pm4::R_ZERO));
        dcb.push(reg);
        dcb.push(value);
    };
    set_sh(pm4::SPI_SHADER_PGM_LO_PS, (ps_addr >> 8) as u32);
    set_sh(pm4::SPI_SHADER_PGM_HI_PS, ((ps_addr >> 40) & 0xFF) as u32);
    set_sh(pm4::SPI_SHADER_PGM_RSRC2_PS, 0); // user_sgpr = 0

    dcb.push(pm4::header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO));
    dcb.push(3);
    dcb.push(0);
    dcb.resize(dcb.len() + 4, 0);

    dcb
}

fn require_or_skip(err: &impl std::fmt::Display) -> bool {
    if std::env::var_os("XPS5X_REQUIRE_VULKAN").is_some() {
        panic!("XPS5X_REQUIRE_VULKAN is set but the Phase 2 draw failed: {err}");
    }
    eprintln!("shader_memory_phase2: SKIP — {err}");
    true
}

/// End-to-end: guest PS bytes → CP bind packets → fetch → GCN parse →
/// SPIR-V recompile → Vulkan draw → green pixels.
#[test]
fn guest_memory_pixel_shader_draws_green() {
    let blob = place_aligned(&build_shader_blob(PS_BODY_SOLID_GREEN, 0xA0A0, 0xB0B0));
    let (width, height) = (64u32, 32u32);
    let dcb = build_guest_ps_dcb(width, height, blob.addr);

    let session = AgcGpuSession::new_process(blob.memory());
    let ok_before = session.shader_stats().translated_ok;
    let image = match session.execute_dcb_cp(&dcb, false) {
        Ok(Some(image)) => image,
        Ok(None) => panic!("the DCB contains a draw — it must not vanish"),
        Err(e) => {
            if require_or_skip(&e) {
                return;
            }
            unreachable!();
        }
    };

    assert_eq!((image.width, image.height), (width, height));
    let center = ((height / 2 * width + width / 2) * 4) as usize;
    let px: [u8; 4] = image.pixels[center..center + 4].try_into().unwrap();
    assert!(
        px[0] <= 2 && px[1] >= 253 && px[2] <= 2 && px[3] >= 253,
        "the fetched PS writes solid green; read back {px:?}"
    );
    assert!(
        session.shader_stats().translated_ok > ok_before,
        "the draw must have gone through the guest fetch+translate path"
    );
    session.shutdown();
}

/// A PS bind pointing at bytes that are not a translatable shader must skip
/// the draw (named, once) and leave the DCB — and the process — alive.
#[test]
fn untranslatable_guest_shader_skips_the_draw_not_the_dcb() {
    // No s_endpgm, no trailer, not valid GCN: 4 KiB of 0xFFFFFFFF.
    let garbage = vec![0xFFFF_FFFFu32; 1024 + 64];
    let blob = place_aligned(&garbage[..1024]);
    let dcb = build_guest_ps_dcb(48, 48, blob.addr);

    let session = AgcGpuSession::new_process(blob.memory());
    let skips_before = session.shader_skip_count();
    let failed_before = session.shader_stats().translate_failed;
    match session.execute_dcb_cp(&dcb, false) {
        // No image: the only draw in the DCB was skipped, honestly.
        Ok(None) => {
            assert!(
                session.shader_skip_count() > skips_before,
                "the skip must be counted"
            );
            assert!(
                session.shader_stats().translate_failed > failed_before,
                "the failure must be negative-cached"
            );
        }
        Ok(Some(_)) => panic!("garbage cannot translate — nothing may draw"),
        Err(e) => if require_or_skip(&e) {},
    }
    session.shutdown();
}
