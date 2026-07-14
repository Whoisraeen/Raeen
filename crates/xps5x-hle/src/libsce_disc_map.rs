//! HLE libSceDiscMap — disc/HDD content-location queries.
//!
//! A disc-based title asks whether a requested content range lives on the
//! HDD (already installed) versus the disc (must be streamed). XPS5X runs
//! against fully-local content, so every request reports **on HDD** — the
//! title treats its data as installed and reads it directly. Behavior and
//! the two undocumented `sceDiscMapUnknown*` mapping-triple writers are
//! cross-checked against SharpEmu's `DiscMapExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_DISC_MAP_ERROR_INVALID_ARGUMENT`.
const ERROR_INVALID_ARGUMENT: u64 = 0x8141_0002;

/// Register libSceDiscMap HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceDiscMap",
        "sceDiscMapIsRequestOnHDD",
        hle_is_request_on_hdd,
    );
    registry.register("libSceDiscMap", "sceDiscMapUnknownFJgP", hle_mapping_triple);
    registry.register("libSceDiscMap", "sceDiscMapUnknownIoKM", hle_mapping_triple);
}

/// `sceDiscMapIsRequestOnHDD(path, offset, size, int *result)`: reports the
/// requested range as present on the HDD (`*result = 1`) — content is local.
fn hle_is_request_on_hdd(ctx: &HleContext, args: &[u64]) -> u64 {
    let path = args.first().copied().unwrap_or(0);
    let result_ptr = args.get(3).copied().unwrap_or(0);
    debug!("sceDiscMapIsRequestOnHDD(path={path:#x}, result={result_ptr:#x}) -> on HDD");
    if path == 0 || result_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(result_ptr, &1i32.to_le_bytes()) {
        warn!("sceDiscMapIsRequestOnHDD: result out-ptr {result_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// The undocumented `sceDiscMapUnknownFJgP`/`IoKM` (mapping-triple writers):
/// `(path, u64 *flags, u64 *out1, u64 *out2)` — writes zeros (no special
/// mapping), matching SharpEmu.
fn hle_mapping_triple(ctx: &HleContext, args: &[u64]) -> u64 {
    let path = args.first().copied().unwrap_or(0);
    let flags = args.get(1).copied().unwrap_or(0);
    let out1 = args.get(2).copied().unwrap_or(0);
    let out2 = args.get(3).copied().unwrap_or(0);
    if path == 0 || flags == 0 || out1 == 0 || out2 == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let ok = ctx.mem.write(flags, &0u64.to_le_bytes())
        && ctx.mem.write(out1, &0u64.to_le_bytes())
        && ctx.mem.write(out2, &0u64.to_le_bytes());
    if !ok {
        warn!("sceDiscMapUnknown*: an out-pointer was not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_ctx, GuestMemory};

    #[test]
    fn request_is_reported_on_hdd() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // path=0x10, offset=0, size=0x100, result=0x200
        assert_eq!(
            hle_is_request_on_hdd(&ctx, &[0x10, 0, 0x100, 0x200]),
            SCE_OK
        );
        let mut r = [0u8; 4];
        assert!(mem.read(0x200, &mut r));
        assert_eq!(i32::from_le_bytes(r), 1, "content is on HDD (local)");
        // NULL result → error.
        assert_eq!(
            hle_is_request_on_hdd(&ctx, &[0x10, 0, 0x100, 0]),
            ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn mapping_triple_zeroes_the_out_params() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        for a in [0x100, 0x108, 0x110] {
            assert!(mem.write(a, &0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()));
        }
        assert_eq!(
            hle_mapping_triple(&ctx, &[0x10, 0x100, 0x108, 0x110]),
            SCE_OK
        );
        for a in [0x100, 0x108, 0x110] {
            let mut v = [0u8; 8];
            assert!(mem.read(a, &mut v));
            assert_eq!(u64::from_le_bytes(v), 0);
        }
    }
}
