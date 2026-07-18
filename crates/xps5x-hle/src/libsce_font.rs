//! HLE libSceFont — font subsystem, starting with memory descriptor init.
//!
//! ASTRO.BOT's ASOBI engine boots its font system in `initFont()` →
//! `fontMemoryCreateByMalloc()` (`FontSystem.cpp:61`), which:
//!   1. `sceLibcMspaceCreate` — carves an 8 MiB font pool, and
//!   2. `sceLibcMspaceCalloc` — allocates a 0x40-byte `OrbisFontMem` descriptor,
//!   3. `sceFontMemoryInit` — binds the pool to that descriptor.
//!
//! With no `libSceFont` provider, step 3 was an unresolved NID: the runtime
//! skipped it, `eax` held garbage-nonzero, the engine read that as an error,
//! asserted, and exited — never reaching the title. `sceFontMemoryInit` here
//! fills the descriptor and returns `ORBIS_OK`, so `fontMemoryCreateByMalloc`
//! succeeds and boot advances.
//!
//! Layout and control flow are ported from shadPS4 (GPL-2.0)
//! `core/libraries/font/font.cpp::sceFontMemoryInit` and its `OrbisFontMem`
//! struct; see `THIRD_PARTY_NOTICES.md`.

use crate::{HleContext, HleRegistry};
use tracing::debug;

/// `ORBIS_OK`.
const ORBIS_OK: u64 = 0;
/// `ORBIS_FONT_ERROR_INVALID_PARAMETER` (`font_error.h`).
const ORBIS_FONT_ERROR_INVALID_PARAMETER: u64 = 0x8046_0002;
/// `ORBIS_FONT_ERROR_NO_SUPPORT_GLYPH` — "this codepoint has no glyph".
const ORBIS_FONT_ERROR_NO_SUPPORT_GLYPH: u64 = 0x8046_0042;

/// `mem_kind` value a live `OrbisFontMem` carries (shadPS4).
const MEM_KIND_LIVE: u16 = 0x0F00;

/// Size of the zeroed guest buffer each opaque font handle (library, renderer,
/// font, selection table) points at. Real libSceFont objects are larger, but
/// nothing in our HLE path dereferences past the header — the handle only has
/// to be a valid, distinct, non-null guest address the title can store and
/// hand back to later (also-HLE'd) font calls.
const HANDLE_BYTES: u64 = 0x100;

/// Allocate a fresh zeroed guest buffer to serve as an opaque font handle, and
/// return its guest address (0 if the arena is exhausted).
fn alloc_handle(ctx: &HleContext) -> u64 {
    let Some(addr) = ctx.alloc.alloc(HANDLE_BYTES, 16) else {
        return 0;
    };
    let _ = ctx.mem.write(addr, &[0u8; HANDLE_BYTES as usize]);
    addr
}

/// Write a freshly-allocated handle to the guest `*out_ptr` and return
/// `ORBIS_OK`. Used by the create/open functions whose last argument is an
/// `OrbisFont** pOut`. A null `out_ptr` or arena exhaustion is a parameter
/// error the title can react to rather than a silent success.
fn return_handle(ctx: &HleContext, out_ptr: u64) -> u64 {
    if out_ptr == 0 {
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }
    let handle = alloc_handle(ctx);
    if handle == 0 {
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }
    if !ctx.mem.write(out_ptr, &handle.to_le_bytes()) {
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }
    ORBIS_OK
}

/// `sceFontMemoryInit(OrbisFontMem* mem, void* region, u32 size,`
/// `const OrbisFontMemInterface* iface, void* mspace,`
/// `OrbisFontMemDestroyCb destroy_cb, void* destroy_ctx)`.
///
/// Initializes the caller-allocated 0x40-byte `OrbisFontMem` descriptor. The
/// game allocates exactly 0x40 bytes for it via `sceLibcMspaceCalloc`, so the
/// field layout below must match shadPS4's byte-for-byte:
///
/// ```text
///   0x00 u16 mem_kind   0x02 u16 attr_bits   0x04 u32 region_size
///   0x08 region_base    0x10 mspace_handle   0x18 iface
///   0x20 on_destroy      0x28 destroy_ctx     0x30 some_ctx1   0x38 some_ctx2
/// ```
fn hle_font_memory_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let mem_desc = args[0];
    let region_addr = args[1];
    let region_size = args[2] as u32;
    let iface = args[3];
    let mspace_obj = args[4];
    let destroy_cb = args[5];
    let destroy_ctx = args.get(6).copied().unwrap_or(0);

    if mem_desc == 0 {
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }
    // Without a custom allocator interface, the caller must supply a real
    // backing region (shadPS4 parity: it zeroes mem_kind and rejects).
    if iface == 0 && (region_addr == 0 || region_size == 0) {
        let _ = ctx.mem.write(mem_desc, &0u16.to_le_bytes());
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }

    let m = ctx.mem;
    let _ = m.write(mem_desc, &MEM_KIND_LIVE.to_le_bytes());
    let _ = m.write(mem_desc + 0x02, &0u16.to_le_bytes());
    let _ = m.write(mem_desc + 0x04, &region_size.to_le_bytes());
    let _ = m.write(mem_desc + 0x08, &region_addr.to_le_bytes());
    let _ = m.write(mem_desc + 0x10, &mspace_obj.to_le_bytes());
    let _ = m.write(mem_desc + 0x18, &iface.to_le_bytes());
    let _ = m.write(mem_desc + 0x20, &destroy_cb.to_le_bytes());
    let _ = m.write(mem_desc + 0x28, &destroy_ctx.to_le_bytes());
    let _ = m.write(mem_desc + 0x30, &0u64.to_le_bytes());
    let _ = m.write(mem_desc + 0x38, &mspace_obj.to_le_bytes());
    debug!(
        "sceFontMemoryInit: mem={mem_desc:#x} region={region_addr:#x} size={region_size:#x} \
         mspace={mspace_obj:#x} iface={iface:#x} -> OK"
    );
    ORBIS_OK
}

/// `sceFontSelectLibraryFt(int value)` / `sceFontSelectRendererFt(int value)`:
/// return a pointer to a driver-selection table (shadPS4 returns a static
/// table for `value == 0`, else null). The title stores the pointer and hands
/// it to `sceFontCreate{Library,Renderer}WithEdition` as create params; our
/// create HLE ignores its contents, so a zeroed handle buffer suffices.
fn hle_select_ft(ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) as u32 != 0 {
        return 0; // non-zero selector -> unsupported (null), matching shadPS4
    }
    alloc_handle(ctx)
}

/// `sceFontCreate{Library,Renderer}WithEdition(memory, params, edition, pOut)`
/// — `pOut` is arg 3.
fn hle_create_with_edition(ctx: &HleContext, args: &[u64]) -> u64 {
    return_handle(ctx, args.get(3).copied().unwrap_or(0))
}

/// `sceFontOpen{FontSet,FontMemory}(..., pFontHandle)` — `pFontHandle` is arg 4.
fn hle_open_font_arg4(ctx: &HleContext, args: &[u64]) -> u64 {
    return_handle(ctx, args.get(4).copied().unwrap_or(0))
}

/// `sceFontOpenFontInstance(fontHandle, setupFont, pFontHandle)` — arg 2 out.
fn hle_open_font_instance(ctx: &HleContext, args: &[u64]) -> u64 {
    return_handle(ctx, args.get(2).copied().unwrap_or(0))
}

/// `sceFontGet{Horizontal,Vertical}Layout(fontHandle, layout*)`: fill the
/// caller's layout struct (arg 1) and report success.
///
/// `OrbisFont{Horizontal,Vertical}Layout` is **exactly 3 floats (12 bytes)**
/// (baseline offset, line/column advance, decoration extent) — and titles pass
/// a *stack-allocated* one. Writing more than 12 bytes overruns it into the
/// caller's stack canary, so the guest's `__stack_chk_fail` fires and traps
/// (this is what crashed ASTRO.BOT after font setup). Write exactly the three
/// fields; use a non-zero advance so text layout that divides by it doesn't
/// hit a divide-by-zero (glyphs still don't rasterize — a later concern).
fn hle_get_layout(ctx: &HleContext, args: &[u64]) -> u64 {
    let layout = args.get(1).copied().unwrap_or(0);
    if layout != 0 {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&0.0f32.to_le_bytes()); // baseline offset
        buf[4..8].copy_from_slice(&1.0f32.to_le_bytes()); // line/column advance
        buf[8..12].copy_from_slice(&0.0f32.to_le_bytes()); // decoration extent
        let _ = ctx.mem.write(layout, &buf);
    }
    ORBIS_OK
}

/// A libSceFont entry point with no meaningful HLE effect that the title only
/// checks the SCE-OK return of. Registered for the wide render/writing/query
/// surface so a UI-text call is a no-op rather than an unresolved jump.
fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ORBIS_OK
}

/// `sceFontRenderCharGlyphImageHorizontal(fontHandle, code, surf, x, y,
/// metrics, result)`. The title renders each glyph into a caller-owned
/// `OrbisFontRenderSurface`, then reads the `OrbisFontRenderOutput` (`result`)
/// to `memcpy` the rendered pixels into its glyph atlas. With no rasterizer we
/// produce a **blank** glyph: point `result->SurfaceImage` at a small zeroed
/// region of the caller's own surface buffer and set a tiny `UpdateRect`, so the
/// copy reads valid (blank) memory instead of faulting on a null bitmap.
///
/// The `float x, y` params sit in XMM registers, so the integer args are
/// `[fontHandle, code, surf, metrics, result]`. Struct offsets are from shadPS4
/// (GPL-2.0) `OrbisFontRenderOutput`/`OrbisFontRenderSurface`/`OrbisFontGlyphMetrics`.
fn hle_render_char_glyph(ctx: &HleContext, args: &[u64]) -> u64 {
    let surf = args.get(2).copied().unwrap_or(0);
    let metrics = args.get(3).copied().unwrap_or(0);
    let result = args.get(4).copied().unwrap_or(0);
    if surf == 0 || result == 0 {
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }
    let m = ctx.mem;
    let ru32 = |a: u64| {
        let mut b = [0u8; 4];
        if m.read(a, &mut b) {
            u32::from_le_bytes(b)
        } else {
            0
        }
    };
    let ru64 = |a: u64| {
        let mut b = [0u8; 8];
        if m.read(a, &mut b) {
            u64::from_le_bytes(b)
        } else {
            0
        }
    };
    let ru8 = |a: u64| {
        let mut b = [0u8; 1];
        m.read(a, &mut b);
        b[0]
    };
    // OrbisFontRenderSurface: 0x00 buffer, 0x08 widthByte, 0x0c pixelSizeByte,
    // 0x10 width, 0x14 height.
    let surf_buffer = ru64(surf);
    let surf_width_byte = ru32(surf + 0x08);
    let surf_pixel_size = u32::from(ru8(surf + 0x0c)).max(1);
    let surf_w = ru32(surf + 0x10);
    let surf_h = ru32(surf + 0x14);
    if surf_buffer == 0 {
        return ORBIS_FONT_ERROR_INVALID_PARAMETER;
    }
    // A tiny blank glyph region, clamped inside the surface.
    let gw = surf_w.clamp(1, 4);
    let gh = surf_h.clamp(1, 4);
    // Zero that region so the copied glyph is blank rather than stale bytes.
    for row in 0..gh {
        let row_addr = surf_buffer + u64::from(row) * u64::from(surf_width_byte);
        let zeros = vec![0u8; (gw * surf_pixel_size) as usize];
        let _ = m.write(row_addr, &zeros);
    }
    // OrbisFontRenderOutput: 0x00 stage, 0x08 SurfaceImage{address, +0x08
    // widthByte, +0x0c pixelSizeByte, +0x0d pixelFormat}, 0x18 UpdateRect{x,y,w,h},
    // 0x28 ImageMetrics{bearingX,bearingY,advance,stride,width,height}.
    let _ = m.write(result, &0u64.to_le_bytes()); // stage = null
    let _ = m.write(result + 0x08, &surf_buffer.to_le_bytes()); // SurfaceImage.address
    let _ = m.write(result + 0x10, &surf_width_byte.to_le_bytes());
    let _ = m.write(result + 0x14, &(surf_pixel_size as u8).to_le_bytes());
    let _ = m.write(result + 0x15, &0u8.to_le_bytes()); // pixelFormat
    let _ = m.write(result + 0x16, &0u16.to_le_bytes()); // pad16
    let _ = m.write(result + 0x18, &0u32.to_le_bytes()); // UpdateRect.x
    let _ = m.write(result + 0x1c, &0u32.to_le_bytes()); // UpdateRect.y
    let _ = m.write(result + 0x20, &gw.to_le_bytes()); // UpdateRect.w
    let _ = m.write(result + 0x24, &gh.to_le_bytes()); // UpdateRect.h
    let _ = m.write(result + 0x28, &0.0f32.to_le_bytes()); // ImageMetrics.bearingX
    let _ = m.write(result + 0x2c, &0.0f32.to_le_bytes()); // bearingY
    let _ = m.write(result + 0x30, &(gw as f32).to_le_bytes()); // advance
    let _ = m.write(result + 0x34, &(gw as f32).to_le_bytes()); // stride
    let _ = m.write(result + 0x38, &gw.to_le_bytes()); // width
    let _ = m.write(result + 0x3c, &gh.to_le_bytes()); // height
    // Caller's OrbisFontGlyphMetrics out (0x20 bytes): width, height, then the
    // Horizontal/Vertical {bearingX,bearingY,advance} sub-structs.
    if metrics != 0 {
        let mut mb = [0u8; 0x20];
        mb[0x00..0x04].copy_from_slice(&(gw as f32).to_le_bytes()); // width
        mb[0x04..0x08].copy_from_slice(&(gh as f32).to_le_bytes()); // height
        mb[0x10..0x14].copy_from_slice(&(gw as f32).to_le_bytes()); // Horizontal.advance
        let _ = m.write(metrics, &mb);
    }
    ORBIS_OK
}

/// `sceFontGenerateCharGlyph(handle, codepoint, params, OrbisFontGlyph* out)`:
/// with no rasterizer we cannot produce a glyph. Initialize the caller's glyph
/// pointer to null (as shadPS4 does at entry) and report the codepoint as
/// unsupported, so the title's text renderer skips it instead of `memcpy`-ing a
/// bitmap out of a never-written (garbage/null) glyph — which faulted on a null
/// source. Text simply doesn't rasterize yet; the engine keeps running.
fn hle_generate_glyph(ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(&out) = args.get(3) {
        if out != 0 {
            let _ = ctx.mem.write(out, &0u64.to_le_bytes());
        }
    }
    ORBIS_FONT_ERROR_NO_SUPPORT_GLYPH
}

/// Register libSceFont / libSceFontFt HLE functions.
///
/// ASTRO.BOT imports ~54 of these. The create/open/select set returns valid
/// opaque handles so `initFont()` completes; the render/writing/query set
/// returns `ORBIS_OK` (a no-op) so later UI-text calls neither crash nor block
/// — text simply doesn't rasterize yet (full glyph rendering is later work).
/// Semantics/arg positions are from shadPS4 (GPL-2.0) `font.cpp`/`fontft.cpp`.
pub fn register(registry: &HleRegistry) {
    // libSceFontFt driver/renderer selection (returns a table pointer).
    registry.register("libSceFontFt", "sceFontSelectLibraryFt", hle_select_ft);
    registry.register("libSceFontFt", "sceFontSelectRendererFt", hle_select_ft);

    // Memory + library/renderer/font lifecycle (return handles).
    registry.register("libSceFont", "sceFontMemoryInit", hle_font_memory_init);
    registry.register(
        "libSceFont",
        "sceFontCreateLibraryWithEdition",
        hle_create_with_edition,
    );
    registry.register(
        "libSceFont",
        "sceFontCreateRendererWithEdition",
        hle_create_with_edition,
    );
    registry.register("libSceFont", "sceFontOpenFontSet", hle_open_font_arg4);
    registry.register("libSceFont", "sceFontOpenFontMemory", hle_open_font_arg4);
    registry.register(
        "libSceFont",
        "sceFontOpenFontInstance",
        hle_open_font_instance,
    );

    // Layout queries (zero the out struct).
    registry.register("libSceFont", "sceFontGetHorizontalLayout", hle_get_layout);
    registry.register("libSceFont", "sceFontGetVerticalLayout", hle_get_layout);

    // Glyph generation: no rasterizer, so report the codepoint unsupported and
    // null the out-glyph so the title skips rendering it.
    registry.register("libSceFont", "sceFontGenerateCharGlyph", hle_generate_glyph);
    // Glyph rendering: point the render output at a blank region of the
    // caller's own surface so its atlas-upload memcpy reads valid memory.
    registry.register(
        "libSceFont",
        "sceFontRenderCharGlyphImageHorizontal",
        hle_render_char_glyph,
    );

    // Everything else the title imports: a checked no-op returning ORBIS_OK.
    // (Setup/effect/bind/render/writing/character/string/support/close/destroy.)
    for func in [
        "sceFontAttachDeviceCacheBuffer",
        "sceFontBindRenderer",
        "sceFontUnbindRenderer",
        "sceFontCharacterGetBidiLevel",
        "sceFontCharacterGetTextFontCode",
        "sceFontCharacterGetTextOrder",
        "sceFontCharacterLooksWhiteSpace",
        "sceFontCharacterRefersTextNext",
        "sceFontCloseFont",
        "sceFontCreateString",
        "sceFontCreateWritingLine",
        "sceFontDeleteGlyph",
        "sceFontDestroyRenderer",
        "sceFontDestroyString",
        "sceFontDestroyWritingLine",
        "sceFontGetRenderCharGlyphMetrics",
        "sceFontGlyphDefineAttribute",
        "sceFontRenderSurfaceInit",
        "sceFontSetEffectSlant",
        "sceFontSetEffectWeight",
        "sceFontSetScalePixel",
        "sceFontSetupRenderEffectSlant",
        "sceFontSetupRenderEffectWeight",
        "sceFontSetupRenderScalePixel",
        "sceFontStringGetTerminateCode",
        "sceFontStringGetWritingForm",
        "sceFontStringRefersRenderCharacters",
        "sceFontStringRefersTextCharacters",
        "sceFontSupportExternalFonts",
        "sceFontSupportSystemFonts",
        "sceFontTextSourceInit",
        "sceFontTextSourceSetDefaultFont",
        "sceFontTextSourceSetWritingForm",
        "sceFontWritingGetRenderMetrics",
        "sceFontWritingInit",
        "sceFontWritingLineClear",
        "sceFontWritingLineGetOrderingSpace",
        "sceFontWritingLineGetRenderMetrics",
        "sceFontWritingLineRefersRenderStep",
        "sceFontWritingLineWritesOrder",
        "sceFontWritingRefersRenderStep",
        "sceFontWritingRefersRenderStepCharacter",
    ] {
        registry.register("libSceFont", func, hle_ok);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn memory_init_fills_descriptor_and_returns_ok() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let desc = 0x100u64;
        // desc, region, size, iface=0 (region present), mspace, destroy_cb, destroy_ctx
        let args = [desc, 0x900, 0x0080_0000, 0, 0x3_0000_0020, 0xCAFE, 0];
        assert_eq!(hle_font_memory_init(&ctx, &args), ORBIS_OK);

        let mut kind = [0u8; 2];
        assert!(mem.read(desc, &mut kind));
        assert_eq!(u16::from_le_bytes(kind), MEM_KIND_LIVE);
        let mut mspace = [0u8; 8];
        assert!(mem.read(desc + 0x10, &mut mspace));
        assert_eq!(u64::from_le_bytes(mspace), 0x3_0000_0020);
        // some_ctx2 mirrors the mspace handle (shadPS4).
        let mut sc2 = [0u8; 8];
        assert!(mem.read(desc + 0x38, &mut sc2));
        assert_eq!(u64::from_le_bytes(sc2), 0x3_0000_0020);
    }

    #[test]
    fn memory_init_rejects_null_descriptor() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let args = [0u64, 0x900, 0x1000, 0, 0x3_0000_0020, 0, 0];
        assert_eq!(
            hle_font_memory_init(&ctx, &args),
            ORBIS_FONT_ERROR_INVALID_PARAMETER
        );
    }
}
