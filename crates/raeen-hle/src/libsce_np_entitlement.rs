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

    // Consumable-entitlement flow (Tier B, 2026-07-27; measured GTA V
    // imports). Consuming an entitlement is a PSN transaction — offline it
    // refuses with the Np SIGNED_OUT error, so no request ever exists for the
    // poll/abort/delete entry points to find. NOTHING here fabricates a
    // consumed entitlement or a transaction. The lib's own error values are
    // not publicly documented: SIGNED_OUT uses the shared Np space and the
    // not-found path the generic kernel spelling (uncertain codes, marked).
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessRequestConsumeUnifiedEntitlement",
        hle_consume_offline,
    );
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessGenerateTransactionId",
        hle_consume_offline,
    );
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessPollConsumeEntitlement",
        hle_request_not_found,
    );
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessAbortRequest",
        hle_request_not_found,
    );
    registry.register(
        "libSceNpEntitlementAccess",
        "sceNpEntitlementAccessDeleteRequest",
        hle_request_not_found,
    );
}

/// `ORBIS_NP_ERROR_SIGNED_OUT` (shadPS4 `np_error.h`) — the honest offline
/// refusal for PSN transactions.
const NP_ERROR_SIGNED_OUT: u64 = 0x8055_0006;
/// Generic kernel ENOENT — uncertain code (the lib's own not-found value is
/// undocumented); used for request-keyed calls when no request can exist.
const SCE_ERROR_NOT_FOUND: u64 = 0x8002_0002;

/// Consume/transaction entry points refuse offline — starting a PSN
/// entitlement transaction signed out is impossible, and no transaction id is
/// written (the out-struct layout is undocumented, and a fabricated id would
/// be a credential).
fn hle_consume_offline(_ctx: &HleContext, _args: &[u64]) -> u64 {
    tracing::debug!("sceNpEntitlementAccess consume/transaction -> SIGNED_OUT (offline)");
    NP_ERROR_SIGNED_OUT
}

/// Request-keyed calls: creation refuses above, so no request exists.
fn hle_request_not_found(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_ERROR_NOT_FOUND
}

/// `sceNpEntitlementAccessInitialize(initParam, bootParam)`: when `bootParam`
/// is non-null, its 0x20-byte block is zeroed; a fault there is reported.
///
/// The 0x20 size is not established by anything in-tree, and `bootParam` is a
/// block the caller owns — routinely a stack local. Bulk-clearing it therefore
/// goes through the out-buffer guard, which performs the clear only where the
/// block provably is not a caller frame; on a frame it writes nothing rather
/// than risk taking out the caller's neighbouring locals and stack-protector
/// canary over a guessed struct size. Clearing a caller's own parameter block
/// is a courtesy, never an ABI guarantee, so skipping it costs nothing.
fn hle_initialize(ctx: &HleContext, args: &[u64]) -> u64 {
    let boot_param = args.get(1).copied().unwrap_or(0);
    if boot_param != 0
        && !ctx.zero_out_object(
            "libSceNpEntitlementAccess::sceNpEntitlementAccessInitialize",
            boot_param,
            BOOT_PARAM_CLEAR_SIZE,
            0,
        )
    {
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

        // A boot param that lives in the calling thread's stack is a caller
        // frame: the guessed 0x20-byte clear must not touch it (the caller's
        // neighbouring locals and `__stack_chk_guard` canary live there), and
        // the call still succeeds.
        kernel.guest_thread_stacks.insert(1, (0x80, 0x100));
        assert!(mem.write(0x90, &[0xFFu8; BOOT_PARAM_CLEAR_SIZE]));
        assert_eq!(hle_initialize(&ctx, &[0x10, 0x90]), OK);
        let mut frame = [0u8; BOOT_PARAM_CLEAR_SIZE];
        assert!(mem.read(0x90, &mut frame));
        assert!(
            frame.iter().all(|&b| b == 0xFF),
            "a stack-resident bootParam must be left alone"
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
