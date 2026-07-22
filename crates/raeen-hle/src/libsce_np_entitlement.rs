//! HLE libSceNpEntitlementAccess — add-on-content (DLC) entitlement queries.
//!
//! A faithful Rust port of SharpEmu's `NpEntitlementAccessExports` (GPL-2.0).
//! A title initializes the library (optionally zeroing a boot-param block) and
//! queries the list of owned add-on-content entitlements. Raeen models a title
//! that owns **no** DLC entitlements: `GetAddcontEntitlementInfoList` writes an
//! empty (zeroed) list header and reports success, so the title sees "no DLC
//! owned" rather than an error. This is faithful bookkeeping, not a real
//! entitlement backend.
//!
//! The generic `OrbisGen2Result` codes are mapped to the real Orbis
//! `OK`/`EFAULT` values as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};

const OK: u64 = 0;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

/// Bytes of the boot-param block cleared by `Initialize`.
const BOOT_PARAM_CLEAR_SIZE: usize = 0x20;
/// Bytes of the empty add-on-content info list header.
const EMPTY_ADDCONT_INFO_LIST_SIZE: usize = 0x10;
/// `SKU_FLAG_FULL` — the retail (non-trial) SKU, matching AppContent's
/// long-standing 1=trial / 3=full convention.
const SKU_FLAG_FULL: u32 = 3;

/// Register the libSceNpEntitlementAccess functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessInitialize",
        hle_initialize,
    );
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessGetAddcontEntitlementInfoList",
        hle_get_addcont_info_list,
    );
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessGetSkuFlag",
        hle_get_sku_flag,
    );
}

/// `sceNpEntitlementAccessInitialize(initParam, bootParam)`: when `bootParam`
/// is non-null, its 0x20-byte block is zeroed; a fault there is reported.
fn hle_initialize(ctx: &HleContext, args: &[u64]) -> u64 {
    let boot_param = args.get(1).copied().unwrap_or(0);
    if boot_param != 0 && !ctx.mem.write(boot_param, &[0u8; BOOT_PARAM_CLEAR_SIZE]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceNpEntitlementAccessGetAddcontEntitlementInfoList(service, list, max,
/// flags)`: writes an empty (zeroed) list header at `list` (arg 1) when
/// non-null — the guest sees no owned add-on content.
fn hle_get_addcont_info_list(ctx: &HleContext, args: &[u64]) -> u64 {
    let list = args.get(1).copied().unwrap_or(0);
    if list != 0 && !ctx.mem.write(list, &[0u8; EMPTY_ADDCONT_INFO_LIST_SIZE]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceNpEntitlementAccessGetSkuFlag(SceNpEntitlementAccessSkuFlag* flag)`:
/// reports the retail SKU. Minecraft's main thread calls this right before
/// bringing up its menu and treats a missing answer as fatal.
fn hle_get_sku_flag(ctx: &HleContext, args: &[u64]) -> u64 {
    let flag = args.first().copied().unwrap_or(0);
    if flag == 0 || !ctx.mem.write(flag, &SKU_FLAG_FULL.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn initialize_clears_boot_param() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Pre-dirty the boot param block.
        assert!(mem.write(0x40, &[0xFFu8; BOOT_PARAM_CLEAR_SIZE]));
        assert_eq!(hle_initialize(&ctx, &[0x10, 0x40]), OK);
        let mut buf = [0xAAu8; BOOT_PARAM_CLEAR_SIZE];
        assert!(mem.read(0x40, &mut buf));
        assert!(buf.iter().all(|&b| b == 0));
        // Null boot param is fine.
        assert_eq!(hle_initialize(&ctx, &[0x10, 0]), OK);
        // Unwritable boot param faults.
        assert_eq!(
            hle_initialize(&ctx, &[0x10, 0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn get_sku_flag_reports_full_sku() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_get_sku_flag(&ctx, &[0x60]), OK);
        let mut buf = [0u8; 4];
        assert!(mem.read(0x60, &mut buf));
        assert_eq!(u32::from_le_bytes(buf), SKU_FLAG_FULL);
        assert_eq!(hle_get_sku_flag(&ctx, &[0]), SCE_ERROR_MEMORY_FAULT);

        let registry = HleRegistry::new();
        register(&registry);
        assert!(registry.registered_names().iter().any(|(library, name)| {
            library == "libSceNpEntitlementAccess" && name == "sceNpEntitlementAccessGetSkuFlag"
        }));
    }

    #[test]
    fn get_addcont_info_list_writes_empty_header() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x50, &[0xFFu8; EMPTY_ADDCONT_INFO_LIST_SIZE]));
        // args: service, list, max, flags.
        assert_eq!(hle_get_addcont_info_list(&ctx, &[1, 0x50, 8, 0]), OK);
        let mut buf = [0xAAu8; EMPTY_ADDCONT_INFO_LIST_SIZE];
        assert!(mem.read(0x50, &mut buf));
        assert!(buf.iter().all(|&b| b == 0));
        // Null list pointer is a benign success.
        assert_eq!(hle_get_addcont_info_list(&ctx, &[1, 0, 8, 0]), OK);
    }
}
