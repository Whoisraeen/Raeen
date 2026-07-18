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

/// `mem_kind` value a live `OrbisFontMem` carries (shadPS4).
const MEM_KIND_LIVE: u16 = 0x0F00;

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

/// Register libSceFont HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceFont", "sceFontMemoryInit", hle_font_memory_init);
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
