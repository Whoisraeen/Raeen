//! HLE libSceNet / libSceNetCtl — network interface (offline).
//!
//! Raeen models **no network connection**: `sceNetCtlGetState` reports
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
/// `SCE_NET_CTL_ERROR_INVALID_ADDR`.
const NET_CTL_ERROR_INVALID_ADDRESS: u64 = 0x8041_2107;
/// `SCE_NET_CTL_ERROR_NOT_CONNECTED`.
const NET_CTL_ERROR_NOT_CONNECTED: u64 = 0x8041_2108;
/// `SCE_NET_CTL_STATE_DISCONNECTED` (0). (`CONNECTING = 1`, ...,
/// `IPOBTAINED = 3`.)
const NET_CTL_STATE_DISCONNECTED: u32 = 0;
/// Monotonic id counter for pool/resolver handles (must be positive).
static NEXT_NET_ID: AtomicU32 = AtomicU32::new(1);

/// Register libSceNet + libSceNetCtl HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceNet", "sceNetInit", hle_ok);
    registry.register("libSceNet", "sceNetTerm", hle_ok);
    // libSceRudp (reliable-UDP P2P transport) lives beside the socket layer.
    // Only the library init is modelled: it succeeds so a title's network stack
    // finishes coming up, while the actual peer-to-peer calls stay unimplemented
    // and will name themselves if a title reaches them. shadPS4 stubs
    // `sceRudpInit` the same way (`rudp.cpp:96`, NID `amuBfI-AQc4`).
    // Measured: ASTRO.BOT stops its boot here once it clears AGC init.
    registry.register("libSceRudp", "sceRudpInit", hle_ok);
    // Accepting the event handler registration keeps the title's network setup
    // moving; with no peer transport there is simply never an event to deliver
    // through it, which is the honest offline behaviour.
    registry.register("libSceRudp", "sceRudpSetEventHandler", hle_ok);
    // Accepting the internal I/O thread request without starting one is honest:
    // there is no peer transport for it to service.
    registry.register("libSceRudp", "sceRudpEnableInternalIOThread", hle_ok);
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
    registry.register("libSceNet", "sceNetInetPton", hle_inet_pton);

    // -- sceNet socket layer (offline) --
    // The sce-prefixed face of the same offline sockets `kernel_socket` models
    // for the POSIX spellings, drawing from the same descriptor pool. The
    // difference is the error convention: SCE_NET range codes (0x8041_01xx)
    // returned directly, not POSIX -1 + errno. Measured: the Minecraft title's
    // network threads import the whole family and would otherwise jump to the
    // unresolved-import stub.
    registry.register("libSceNet", "sceNetSocket", hle_net_socket);
    registry.register("libSceNet", "sceNetSocketClose", hle_net_socket_close);
    registry.register("libSceNet", "sceNetBind", hle_net_bind);
    registry.register("libSceNet", "sceNetConnect", hle_net_connect);
    registry.register("libSceNet", "sceNetListen", hle_net_listen);
    registry.register("libSceNet", "sceNetAccept", hle_net_accept);
    registry.register("libSceNet", "sceNetSend", hle_net_send);
    registry.register("libSceNet", "sceNetRecv", hle_net_recv);
    registry.register("libSceNet", "sceNetShutdown", hle_net_shutdown);
    registry.register("libSceNet", "sceNetSetsockopt", hle_net_setsockopt);
    registry.register("libSceNet", "sceNetErrnoLoc", hle_net_errno_loc);
    registry.register(
        "libSceNet",
        "sceNetResolverStartNtoa",
        hle_net_resolver_start_ntoa,
    );

    registry.register("libSceNetCtl", "sceNetCtlInit", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlTerm", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlGetState", hle_ctl_get_state);
    registry.register("libSceNetCtl", "sceNetCtlGetStateV6", hle_ctl_get_state);
    registry.register("libSceNetCtl", "sceNetCtlCheckCallback", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlRegisterCallback", hle_ok);
    registry.register("libSceNetCtl", "sceNetCtlRegisterCallbackV6", hle_ok);
    registry.register("libSceNet", "sceNetEpollCreate", hle_epoll_create);
    registry.register("libSceNet", "sceNetEpollControl", hle_epoll_control);
    registry.register("libSceNet", "sceNetEpollWait", hle_epoll_wait);
    registry.register("libSceNet", "sceNetEpollDestroy", hle_epoll_destroy);
    registry.register("libSceNetCtl", "sceNetCtlGetInfo", hle_ctl_get_info);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `SCE_NET_ERROR_EBADF` — unknown socket fd.
const NET_ERROR_EBADF: u64 = 0x8041_0109;
/// `SCE_NET_ERROR_EFAULT` — unreadable guest payload/address.
const NET_ERROR_EFAULT: u64 = 0x8041_010e;
/// `SCE_NET_ERROR_EMFILE` — descriptor table full.
const NET_ERROR_EMFILE: u64 = 0x8041_0118;
/// `SCE_NET_ERROR_EWOULDBLOCK` — empty accept backlog / no data (offline).
const NET_ERROR_EWOULDBLOCK: u64 = 0x8041_0123;
/// `SCE_NET_ERROR_ENETUNREACH` — no route: Raeen models no link at all.
const NET_ERROR_ENETUNREACH: u64 = 0x8041_0133;
/// `SCE_NET_ERROR_RESOLVER_ENODNS` — no DNS offline (shadPS4's offline
/// answer for `sceNetResolverStartNtoa` too).
const NET_ERROR_RESOLVER_ENODNS: u64 = 0x8041_01e1;

/// `sceNetSocket(domain, type, protocol)`: a fresh offline socket fd from the
/// same pool the POSIX `socket` spelling uses.
fn hle_net_socket(ctx: &HleContext, _args: &[u64]) -> u64 {
    let Some(fd) = ctx.services.create_socket() else {
        return NET_ERROR_EMFILE;
    };
    debug!("sceNetSocket() -> offline fd {fd:#x}");
    fd as u32 as u64
}

/// `sceNetSocketClose(fd)`: drop the offline socket.
fn hle_net_socket_close(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if ctx.services.close_socket(fd) {
        SCE_OK
    } else {
        NET_ERROR_EBADF
    }
}

/// `sceNetBind(fd, addr, addrlen)`: record the bound address (offline) — the
/// POSIX `bind` behavior with the sce error convention.
fn hle_net_bind(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return NET_ERROR_EBADF;
    }
    if crate::kernel_socket::bind_offline(ctx, args) {
        SCE_OK
    } else {
        NET_ERROR_INVALID_ARGUMENT
    }
}

/// `sceNetConnect(fd, addr, addrlen)`: no host connectivity — the honest
/// offline answer is ENETUNREACH (NetCtl already reports DISCONNECTED).
fn hle_net_connect(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return NET_ERROR_EBADF;
    }
    debug!("sceNetConnect(fd={fd:#x}) -> ENETUNREACH (offline)");
    NET_ERROR_ENETUNREACH
}

/// `sceNetListen(fd, backlog)`: accepted; an offline listener simply never
/// becomes ready.
fn hle_net_listen(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if ctx.services.socket_exists(fd) {
        SCE_OK
    } else {
        NET_ERROR_EBADF
    }
}

/// `sceNetAccept(fd, addr, addrlen)`: no peer can ever connect — report an
/// empty backlog (`EWOULDBLOCK`) rather than blocking forever.
fn hle_net_accept(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return NET_ERROR_EBADF;
    }
    crate::libkernel::set_guest_errno(ctx, 35); // EWOULDBLOCK
    NET_ERROR_EWOULDBLOCK
}

/// `sceNetSend(fd, buf, len, flags)`: bytes into the void, exactly like the
/// POSIX `send` spelling — validated as readable guest memory first.
fn hle_net_send(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return NET_ERROR_EBADF;
    }
    crate::kernel_socket::send_offline(ctx, args).unwrap_or(NET_ERROR_EFAULT)
}

/// `sceNetRecv(fd, buf, len, flags)`: no data can ever arrive — `EWOULDBLOCK`.
fn hle_net_recv(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return NET_ERROR_EBADF;
    }
    crate::kernel_socket::backoff_offline_recv(ctx);
    crate::libkernel::set_guest_errno(ctx, 35); // EWOULDBLOCK
    NET_ERROR_EWOULDBLOCK
}

/// `sceNetShutdown(fd, how)`: nothing is connected, so both directions are
/// already shut; accept the call.
fn hle_net_shutdown(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if ctx.services.socket_exists(fd) {
        SCE_OK
    } else {
        NET_ERROR_EBADF
    }
}

/// `sceNetSetsockopt(fd, level, optname, optval, optlen)`: validate and accept
/// — no option can enable host connectivity.
fn hle_net_setsockopt(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return NET_ERROR_EBADF;
    }
    if crate::kernel_socket::setsockopt_offline(ctx, args) {
        SCE_OK
    } else {
        NET_ERROR_INVALID_ARGUMENT
    }
}

/// `sceNetErrnoLoc()`: address of the calling thread's net errno cell. Shared
/// with `__error()` so `*sceNetErrnoLoc()` reads back whatever the last
/// failing POSIX/sceNet call recorded.
fn hle_net_errno_loc(ctx: &HleContext, _args: &[u64]) -> u64 {
    crate::libkernel::hle_error_addr(ctx, &[])
}

/// `sceNetResolverStartNtoa(rid, hostname, addr, timeout, retry, flags)`:
/// offline there is no DNS — fail the way a disconnected console does
/// (`SCE_NET_ERROR_RESOLVER_ENODNS`).
fn hle_net_resolver_start_ntoa(_ctx: &HleContext, args: &[u64]) -> u64 {
    let rid = args.first().copied().unwrap_or(0);
    let hostname = args.get(1).copied().unwrap_or(0);
    debug!("sceNetResolverStartNtoa(rid={rid}, hostname={hostname:#x}) -> ENODNS (offline)");
    NET_ERROR_RESOLVER_ENODNS
}

/// Hand back a fresh positive handle (pool / resolver).
fn hle_new_id(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NEXT_NET_ID.fetch_add(1, Ordering::Relaxed) as u64
}

/// Real `sceNetInetPton(af, const char* src, void* dst)`: parse a textual
/// address into network-order bytes. `SCE_NET_AF_INET` = 2 (4 bytes out),
/// `SCE_NET_AF_INET6` = 28 (16 bytes out). Returns 1 on success, 0 for an
/// unparseable string (matching POSIX inet_pton), EINVAL for a bad family.
fn hle_inet_pton(ctx: &HleContext, args: &[u64]) -> u64 {
    let family = args.first().copied().unwrap_or(0) as u32;
    let src = args.get(1).copied().unwrap_or(0);
    let dst = args.get(2).copied().unwrap_or(0);
    if src == 0 || dst == 0 {
        return NET_ERROR_INVALID_ARGUMENT;
    }
    let Some(text) = crate::fmt::read_cstr(ctx.mem, src) else {
        return NET_ERROR_INVALID_ARGUMENT;
    };
    let Ok(text) = std::str::from_utf8(&text) else {
        return 0;
    };
    match family {
        2 => match text.parse::<std::net::Ipv4Addr>() {
            Ok(addr) if ctx.mem.write(dst, &addr.octets()) => 1,
            Ok(_) => NET_ERROR_INVALID_ARGUMENT,
            Err(_) => 0,
        },
        28 => match text.parse::<std::net::Ipv6Addr>() {
            Ok(addr) if ctx.mem.write(dst, &addr.octets()) => 1,
            Ok(_) => NET_ERROR_INVALID_ARGUMENT,
            Err(_) => 0,
        },
        _ => NET_ERROR_INVALID_ARGUMENT,
    }
}

/// Real `sceNetHtonl(uint32_t)`: host→network (big-endian) byte order.
fn hle_htonl(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.first().copied().unwrap_or(0) as u32;
    v.to_be() as u64
}

/// `sceNetEpollCreate(name, flags)`: a fresh offline epoll id (empty set).
fn hle_epoll_create(ctx: &HleContext, _args: &[u64]) -> u64 {
    let id = ctx.kernel.create_epoll();
    debug!("sceNetEpollCreate() -> {id}");
    u64::from(id)
}

/// `sceNetEpollControl(epid, op, fd, event)`: ADD=1 / MOD=2 records
/// (fd, events, data); DEL=3 drops the fd. Offline sockets only.
fn hle_epoll_control(ctx: &HleContext, args: &[u64]) -> u64 {
    const NET_ERROR_ENOENT: u64 = 0x8041_0103;
    let epid = args.first().copied().unwrap_or(0) as u32;
    let op = args.get(1).copied().unwrap_or(0);
    let fd = args.get(2).copied().unwrap_or(0) as i32;
    let event_ptr = args.get(3).copied().unwrap_or(0);
    let Some(mut set) = ctx.kernel.kernel_epolls.get_mut(&epid) else {
        return NET_ERROR_ENOENT;
    };
    match op {
        1 | 2 => {
            // SceNetEpollEvent: events u32 @0, data u64 @8.
            let mut raw = [0u8; 16];
            if event_ptr != 0 && !ctx.mem.read(event_ptr, &mut raw) {
                return NET_ERROR_INVALID_ARGUMENT;
            }
            let events = u32::from_le_bytes(raw[0..4].try_into().expect("fixed slice"));
            let data = u64::from_le_bytes(raw[8..16].try_into().expect("fixed slice"));
            set.retain(|(f, _, _)| *f != fd);
            set.push((fd, events, data));
        }
        3 => set.retain(|(f, _, _)| *f != fd),
        _ => return NET_ERROR_INVALID_ARGUMENT,
    }
    SCE_OK
}

/// `sceNetEpollWait(epid, events, maxevents, timeout)`: offline — no fd ever
/// becomes ready. Wait a bounded slice of the requested timeout (an unbounded
/// block would hang the caller; the equeue lesson) and report 0 events.
fn hle_epoll_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let epid = args.first().copied().unwrap_or(0) as u32;
    let timeout_ms = args.get(3).copied().unwrap_or(0) as i32;
    if !ctx.kernel.kernel_epolls.contains_key(&epid) {
        return NET_ERROR_INVALID_ARGUMENT;
    }
    let wait = if timeout_ms < 0 {
        50
    } else {
        (timeout_ms as u64).min(50)
    };
    std::thread::sleep(std::time::Duration::from_millis(wait));
    0
}

/// `sceNetEpollDestroy(epid)`: drop the set.
fn hle_epoll_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let epid = args.first().copied().unwrap_or(0) as u32;
    ctx.kernel.kernel_epolls.remove(&epid);
    SCE_OK
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

/// `sceNetCtlGetInfo(code, info)`: report a deterministic offline interface.
/// The tagged union's active field is selected by `code`; fixed-size string
/// members are cleared before their loopback/offline value is copied.
fn hle_ctl_get_info(ctx: &HleContext, args: &[u64]) -> u64 {
    let code = args.first().copied().unwrap_or(0) as i32;
    let info = args.get(1).copied().unwrap_or(0);
    if info == 0 {
        return NET_CTL_ERROR_INVALID_ADDRESS;
    }

    let result = match code {
        1 | 4 | 11 | 19 => ctx.mem.write(info, &0u32.to_le_bytes()),
        2 => ctx.mem.write(info, &[0u8; 6]),
        3 => ctx.mem.write(info, &1500u32.to_le_bytes()),
        12 => write_fixed_string(ctx, info, 256, ""),
        13 => write_fixed_string(ctx, info, 128, ""),
        14 | 16 => write_fixed_string(ctx, info, 16, "127.0.0.1"),
        15 => write_fixed_string(ctx, info, 16, "255.0.0.0"),
        17 | 18 => write_fixed_string(ctx, info, 16, "1.1.1.1"),
        20 => write_fixed_string(ctx, info, 256, ""),
        21 => ctx.mem.write(info, &0u16.to_le_bytes()),
        _ => return NET_CTL_ERROR_NOT_CONNECTED,
    };
    if result {
        SCE_OK
    } else {
        NET_CTL_ERROR_INVALID_ADDRESS
    }
}

fn write_fixed_string(ctx: &HleContext, address: u64, size: usize, value: &str) -> bool {
    let mut bytes = vec![0u8; size];
    let count = value.len().min(size.saturating_sub(1));
    bytes[..count].copy_from_slice(&value.as_bytes()[..count]);
    ctx.mem.write(address, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn scenet_socket_family_is_offline_consistent() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let fd = hle_net_socket(&ctx, &[2, 1, 0]);
        assert!(fd < 0x8000_0000, "sceNetSocket must yield a small fd");
        assert_eq!(hle_net_connect(&ctx, &[fd, 0, 0]), NET_ERROR_ENETUNREACH);
        assert_eq!(hle_net_accept(&ctx, &[fd]), NET_ERROR_EWOULDBLOCK);
        assert_eq!(hle_net_recv(&ctx, &[fd, 0, 0, 0]), NET_ERROR_EWOULDBLOCK);
        assert_eq!(hle_net_listen(&ctx, &[fd, 4]), SCE_OK);
        assert_eq!(hle_net_shutdown(&ctx, &[fd, 2]), SCE_OK);

        // send discards a validated payload and reports it "sent" (offline).
        assert!(mem.write(0x40, b"ping"));
        assert_eq!(hle_net_send(&ctx, &[fd, 0x40, 4, 0]), 4);

        assert_eq!(hle_net_socket_close(&ctx, &[fd]), SCE_OK);
        assert_eq!(hle_net_socket_close(&ctx, &[fd]), NET_ERROR_EBADF);
        assert_eq!(hle_net_connect(&ctx, &[0x7777]), NET_ERROR_EBADF);
        assert_eq!(hle_net_send(&ctx, &[0x7777, 0x40, 4, 0]), NET_ERROR_EBADF);
    }

    #[test]
    fn scenet_resolver_and_errno_loc_report_offline() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10000);
        let alloc = crate::TestAllocator::new(0x8000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_net_resolver_start_ntoa(&ctx, &[1, 0, 0, 0, 0, 0]),
            NET_ERROR_RESOLVER_ENODNS
        );
        // ErrnoLoc shares the __error cell: stable, nonzero, and it reads back
        // the EWOULDBLOCK a failed sceNetAccept recorded.
        let fd = hle_net_socket(&ctx, &[2, 1, 0]);
        assert_eq!(hle_net_accept(&ctx, &[fd]), NET_ERROR_EWOULDBLOCK);
        let slot = hle_net_errno_loc(&ctx, &[]);
        assert_ne!(slot, 0);
        assert_eq!(hle_net_errno_loc(&ctx, &[]), slot);
        let mut raw = [0u8; 4];
        assert!(mem.read(slot, &mut raw));
        assert_eq!(i32::from_le_bytes(raw), 35);
    }

    #[test]
    fn scenet_socket_family_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceNetSocket",
            "sceNetSocketClose",
            "sceNetBind",
            "sceNetConnect",
            "sceNetListen",
            "sceNetAccept",
            "sceNetSend",
            "sceNetRecv",
            "sceNetShutdown",
            "sceNetSetsockopt",
            "sceNetErrnoLoc",
            "sceNetResolverStartNtoa",
        ] {
            assert!(
                registry.is_implemented("libSceNet", name),
                "libSceNet::{name} must be registered"
            );
        }
    }

    #[test]
    fn byte_order_helpers_are_real_swaps() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
    fn inet_pton_parses_v4_and_v6_and_rejects_garbage() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x200);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x40, b"192.168.1.20\0"));
        assert_eq!(hle_inet_pton(&ctx, &[2, 0x40, 0x80]), 1);
        let mut v4 = [0u8; 4];
        assert!(mem.read(0x80, &mut v4));
        assert_eq!(v4, [192, 168, 1, 20]);

        assert!(mem.write(0x100, b"::1\0"));
        assert_eq!(hle_inet_pton(&ctx, &[28, 0x100, 0x110]), 1);
        let mut v6 = [0u8; 16];
        assert!(mem.read(0x110, &mut v6));
        assert_eq!(v6, std::net::Ipv6Addr::LOCALHOST.octets());

        assert!(mem.write(0x140, b"not-an-address\0"));
        assert_eq!(hle_inet_pton(&ctx, &[2, 0x140, 0x80]), 0);
        assert_eq!(
            hle_inet_pton(&ctx, &[99, 0x40, 0x80]),
            NET_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_inet_pton(&ctx, &[2, 0, 0x80]),
            NET_ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn netctl_reports_disconnected_and_handles_are_positive() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
    fn netctl_get_info_reports_deterministic_offline_values() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_ctl_get_info(&ctx, &[3, 0x40]), SCE_OK);
        let mut mtu = [0u8; 4];
        assert!(mem.read(0x40, &mut mtu));
        assert_eq!(u32::from_le_bytes(mtu), 1500);

        assert_eq!(hle_ctl_get_info(&ctx, &[14, 0x80]), SCE_OK);
        let mut address = [0xffu8; 16];
        assert!(mem.read(0x80, &mut address));
        assert_eq!(&address[..10], b"127.0.0.1\0");
        assert!(address[10..].iter().all(|byte| *byte == 0));

        assert_eq!(
            hle_ctl_get_info(&ctx, &[999, 0x40]),
            NET_CTL_ERROR_NOT_CONNECTED
        );
        assert_eq!(
            hle_ctl_get_info(&ctx, &[3, 0]),
            NET_CTL_ERROR_INVALID_ADDRESS
        );

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libSceNetCtl", "sceNetCtlGetInfo"));
    }

    #[test]
    fn get_mac_address_writes_an_offline_zero_address() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
