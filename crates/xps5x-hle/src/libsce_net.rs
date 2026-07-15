//! HLE libSceNet / libSceNetCtl — network interface (offline).
//!
//! XPS5X models **no network connection**: `sceNetCtlGetState` reports
//! `DISCONNECTED`, so an online-aware title sees no link and runs its
//! offline path. `sceNetInit` and the pool/resolver handles succeed so a
//! title's network *initialization* doesn't fail outright at boot (actual
//! connectivity is simply absent). The byte-order helpers
//! (`Htonl`/`Htons`/`Ntohl`/`Ntohs`) are **real** — pure host↔network
//! (big-endian) byte swaps. Export set cross-checked against SharpEmu.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_NET_ERROR_EINVAL`.
const NET_ERROR_INVALID_ARGUMENT: u64 = 0x8041_0116;
/// `SCE_NET_CTL_STATE_DISCONNECTED` (0). (`CONNECTING = 1`, ...,
/// `IPOBTAINED = 3`.)
const NET_CTL_STATE_DISCONNECTED: u32 = 0;
/// Monotonic id counter for pool/resolver handles (must be positive).
static NEXT_NET_ID: AtomicU32 = AtomicU32::new(1);

/// Register libSceNet + libSceNetCtl HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceNet", "sceNetInit", hle_ok);
    registry.register("libSceNet", "sceNetTerm", hle_ok);
    registry.register("libSceNet", "sceNetPoolCreate", hle_new_id);
    registry.register("libSceNet", "sceNetPoolDestroy", hle_ok);
    registry.register("libSceNet", "sceNetResolverCreate", hle_new_id);
    registry.register("libSceNet", "sceNetResolverDestroy", hle_ok);
    registry.register("libSceNet", "sceNetHtonl", hle_htonl);
    registry.register("libSceNet", "sceNetHtons", hle_htons);
    registry.register("libSceNet", "sceNetNtohl", hle_htonl); // symmetric byte swap
    registry.register("libSceNet", "sceNetNtohs", hle_htons);
    registry.register("libSceNet", "sceNetGetMacAddress", hle_get_mac_address);
    registry.register("libSceNet", "sceNetEtherNtostr", hle_ether_ntostr);

    registry.register("libSceNetCtl", "sceNetCtlInit", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlTerm", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlGetState", hle_ctl_get_state);
    registry.register("libSceNetCtl", "sceNetCtlCheckCallback", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlRegisterCallback", hle_ok);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// Hand back a fresh positive handle (pool / resolver).
fn hle_new_id(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NEXT_NET_ID.fetch_add(1, Ordering::Relaxed) as u64
}

/// Real `sceNetHtonl(uint32_t)`: host→network (big-endian) byte order.
fn hle_htonl(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.first().copied().unwrap_or(0) as u32;
    v.to_be() as u64
}

/// Real `sceNetHtons(uint16_t)`: host→network byte order (16-bit).
fn hle_htons(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.first().copied().unwrap_or(0) as u16;
    v.to_be() as u64
}

/// `sceNetGetMacAddress(SceNetEtherAddr *addr, int flags)`: the offline
/// compatibility layer exposes no host hardware identity, so report the
/// conventional all-zero six-byte address. Kyty uses the same behavior.
fn hle_get_mac_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let flags = args.get(1).copied().unwrap_or(0);
    if addr == 0 || flags != 0 || !ctx.mem.write(addr, &[0u8; 6]) {
        return NET_ERROR_INVALID_ARGUMENT;
    }
    debug!("sceNetGetMacAddress(addr={addr:#x}) -> 00:00:00:00:00:00");
    SCE_OK
}

/// `sceNetEtherNtostr(const SceNetEtherAddr *addr, char *str, size_t len)`:
/// render the six-byte Ethernet address as a lower-case, colon-separated
/// string. The ABI requires an 18-byte destination including the terminator.
fn hle_ether_ntostr(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let output = args.get(1).copied().unwrap_or(0);
    let len = args.get(2).copied().unwrap_or(0);
    if addr == 0 || output == 0 || len != 18 {
        return NET_ERROR_INVALID_ARGUMENT;
    }

    let mut mac = [0u8; 6];
    if !ctx.mem.read(addr, &mut mac) {
        return NET_ERROR_INVALID_ARGUMENT;
    }
    let text = format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}\0",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
    );
    if !ctx.mem.write(output, text.as_bytes()) {
        return NET_ERROR_INVALID_ARGUMENT;
    }
    debug!("sceNetEtherNtostr(addr={addr:#x}) -> {}", &text[..17]);
    SCE_OK
}

/// `sceNetCtlGetState(int *state)`: reports `DISCONNECTED` — no network.
fn hle_ctl_get_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let state_ptr = args.first().copied().unwrap_or(0);
    debug!("sceNetCtlGetState(state={state_ptr:#x}) -> DISCONNECTED");
    if state_ptr != 0
        && !ctx
            .mem
            .write(state_ptr, &NET_CTL_STATE_DISCONNECTED.to_le_bytes())
    {
        debug!("sceNetCtlGetState: state out-ptr {state_ptr:#x} not writable");
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn byte_order_helpers_are_real_swaps() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // 0x11223344 -> big-endian 0x44332211 on a little-endian host.
        assert_eq!(hle_htonl(&ctx, &[0x1122_3344]), 0x4433_2211);
        assert_eq!(hle_htons(&ctx, &[0x1122]), 0x2211);
        // htonl ∘ ntohl (same fn) is the identity.
        let round = hle_htonl(&ctx, &[hle_htonl(&ctx, &[0xDEAD_BEEF])]);
        assert_eq!(round, 0xDEAD_BEEF);
    }

    #[test]
    fn netctl_reports_disconnected_and_handles_are_positive() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_ctl_get_state(&ctx, &[0x40]), SCE_OK);
        let mut s = [0u8; 4];
        assert!(mem.read(0x40, &mut s));
        assert_eq!(u32::from_le_bytes(s), NET_CTL_STATE_DISCONNECTED);

        assert!(hle_new_id(&ctx, &[]) > 0);
    }

    #[test]
    fn get_mac_address_writes_an_offline_zero_address() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x40, &[0xAA; 6]));
        assert_eq!(hle_get_mac_address(&ctx, &[0x40, 0]), SCE_OK);
        let mut mac = [0xAA; 6];
        assert!(mem.read(0x40, &mut mac));
        assert_eq!(mac, [0; 6]);
        assert_ne!(hle_get_mac_address(&ctx, &[0, 0]), SCE_OK);
        assert_ne!(hle_get_mac_address(&ctx, &[0x40, 1]), SCE_OK);
    }

    #[test]
    fn ether_ntostr_formats_a_mac_address() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x20, &[0x00, 0x11, 0x22, 0xAA, 0xBB, 0xFF]));
        assert_eq!(hle_ether_ntostr(&ctx, &[0x20, 0x40, 18]), SCE_OK);
        let mut text = [0u8; 18];
        assert!(mem.read(0x40, &mut text));
        assert_eq!(&text, b"00:11:22:aa:bb:ff\0");

        assert_ne!(hle_ether_ntostr(&ctx, &[0, 0x40, 18]), SCE_OK);
        assert_ne!(hle_ether_ntostr(&ctx, &[0x20, 0, 18]), SCE_OK);
        assert_ne!(hle_ether_ntostr(&ctx, &[0x20, 0x40, 17]), SCE_OK);
    }
}
