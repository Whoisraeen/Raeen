//! HLE libkernel **BSD sockets** (offline) + pure network helpers.
//!
//! Ported from SharpEmu's `KernelSocketCompatExports` (GPL-2.0), **minus its
//! real host-TCP path**: Raeen models no host connectivity (consistent with
//! `libSceNetCtl` reporting `DISCONNECTED`), and guest code must never reach
//! the host network. So a socket can be created, bound, and read back via
//! `getsockname`, but `connect` always fails (`-1`).
//!
//! The pure helpers — `htons` (byte swap), `inet_pton` (parse an IPv4 string),
//! `bzero` (zero guest memory) — are **fully correct** and host-independent.
//! Socket state lives in the kernel (`OrbisKernel::kernel_sockets`).

use crate::{HleContext, HleRegistry};
use tracing::debug;

/// BSD sockets return `-1` on error (as a `u64` in the return register).
const MINUS_ONE: u64 = u64::MAX;
const OK: u64 = 0;
const AF_INET: u64 = 2;

/// Register the socket + helper HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "socket", hle_socket);
    registry.register("libkernel", "bind", hle_bind);
    registry.register("libkernel", "connect", hle_connect);
    registry.register("libkernel", "getsockname", hle_getsockname);
    // The same five under the POSIX provider: a NID hashes the function name
    // alone, so libScePosix imports carry these same NIDs and only the
    // provider-aware registration is missing. Measured: the Minecraft title
    // imports `socket` naming libScePosix and died jumping to the
    // unresolved-import stub.
    registry.register("libScePosix", "socket", hle_socket);
    registry.register("libScePosix", "bind", hle_bind);
    registry.register("libScePosix", "connect", hle_connect);
    registry.register("libScePosix", "getsockname", hle_getsockname);
    registry.register("libScePosix", "inet_pton", hle_inet_pton);
    registry.register("libScePosix", "setsockopt", hle_setsockopt);
    registry.register("libkernel", "bzero", hle_bzero);
    registry.register("libkernel", "inet_pton", hle_inet_pton);
    registry.register("libScePosix", "inet_ntop", hle_inet_ntop);
    registry.register("libkernel", "htons", hle_htons);

    // POSIX socket surface the measured title (Minecraft) imports from
    // libScePosix. All offline: nothing ever connects, arrives, or becomes
    // readable — see the per-function notes.
    registry.register("libScePosix", "accept", hle_accept);
    registry.register("libScePosix", "listen", hle_listen);
    registry.register("libScePosix", "recv", hle_recv);
    registry.register("libScePosix", "recvfrom", hle_recv);
    registry.register("libScePosix", "send", hle_send);
    registry.register("libScePosix", "sendto", hle_send);
    registry.register("libScePosix", "shutdown", hle_shutdown);
    registry.register("libScePosix", "getpeername", hle_getpeername);
    registry.register("libScePosix", "getsockopt", hle_getsockopt);
    registry.register("libScePosix", "select", hle_select);
}

/// `EWOULDBLOCK` on FreeBSD/Orbis (35).
const EWOULDBLOCK: i32 = 35;
/// Process descriptor table full.
const EMFILE: i32 = 24;

/// `listen(fd, backlog)`: accept the request for an offline socket. Nothing
/// can ever connect, so a listening socket simply never becomes ready.
fn hle_listen(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    debug!("listen(fd={fd:#x}) -> OK (offline: no peer can ever connect)");
    OK
}

/// `accept(fd, addr, addrlen)`: no host connectivity means no pending
/// connection, ever — report `EWOULDBLOCK` like a non-blocking listener with
/// an empty backlog rather than blocking a guest thread forever.
fn hle_accept(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    debug!("accept(fd={fd:#x}) -> EWOULDBLOCK (offline)");
    crate::libkernel::set_guest_errno(ctx, EWOULDBLOCK);
    MINUS_ONE
}

/// One millisecond keeps an empty offline socket responsive while preventing
/// a guest nonblocking receive loop from monopolizing a host core. Measured on
/// Minecraft's RakThread: the immediate-return path issued 17,883,821
/// `recvfrom` calls in roughly two minutes.
const OFFLINE_RECV_BACKOFF: std::time::Duration = std::time::Duration::from_millis(1);

pub(crate) fn backoff_offline_recv(ctx: &HleContext) {
    if !ctx.guest_threads.process_is_terminating() {
        ctx.services.sleep(OFFLINE_RECV_BACKOFF);
    }
}

/// `recv(fd, buf, len, flags)` / `recvfrom(..., addr, addrlen)`: no data can
/// ever arrive on an offline socket — briefly yield, then return
/// `EWOULDBLOCK`; never invent a payload or park forever on an event that
/// cannot occur.
fn hle_recv(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    backoff_offline_recv(ctx);
    debug!("recv/recvfrom(fd={fd:#x}) -> EWOULDBLOCK (offline)");
    crate::libkernel::set_guest_errno(ctx, EWOULDBLOCK);
    MINUS_ONE
}

/// `send(fd, buf, len, flags)` / `sendto(..., addr, addrlen)`: accept the
/// bytes and report them "sent" (into the void). The payload is validated as
/// readable guest memory so a wild pointer still faults loudly here instead
/// of corrupting later.
fn hle_send(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buf = args.get(1).copied().unwrap_or(0);
    let len = args.get(2).copied().unwrap_or(0);
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    // Bounded validation read: confirm the payload is real guest memory.
    let probe = len.min(4096);
    if probe != 0 {
        let Ok(probe) = usize::try_from(probe) else {
            return MINUS_ONE;
        };
        let mut bytes = vec![0u8; probe];
        if buf == 0 || !ctx.mem.read(buf, &mut bytes) {
            return MINUS_ONE;
        }
    }
    debug!("send/sendto(fd={fd:#x}, len={len:#x}) -> discarded (offline)");
    len
}

/// `shutdown(fd, how)`: nothing is connected, so both directions are already
/// shut; accept the call.
fn hle_shutdown(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    debug!("shutdown(fd={fd:#x}) -> OK (offline)");
    OK
}

/// `getpeername(fd, addr, addrlen)`: an offline socket has no peer; write a
/// zeroed `sockaddr_in` (family only) and report success — callers that only
/// log the peer keep working, and the all-zero address is inert.
fn hle_getpeername(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let sockaddr = args.get(1).copied().unwrap_or(0);
    let addrlen_ptr = args.get(2).copied().unwrap_or(0);
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    let mut lenbuf = [0u8; 4];
    if addrlen_ptr == 0 || !ctx.mem.read(addrlen_ptr, &mut lenbuf) {
        return MINUS_ONE;
    }
    let addrlen = i32::from_le_bytes(lenbuf);
    if addrlen < 8 || sockaddr == 0 {
        return MINUS_ONE;
    }
    let mut sa = [0u8; 16];
    sa[0] = 16;
    sa[1] = AF_INET as u8;
    let write_len = (addrlen as usize).min(16);
    if !ctx.mem.write(sockaddr, &sa[..write_len])
        || !ctx
            .mem
            .write(addrlen_ptr, &(write_len as i32).to_le_bytes())
    {
        return MINUS_ONE;
    }
    OK
}

/// `getsockopt(fd, level, optname, optval, optlen)`: report a zero-filled
/// option value (bounded). Zero is the honest answer for the options titles
/// poll on an offline socket: no error pending (`SO_ERROR`), no bytes ready.
fn hle_getsockopt(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let optval = args.get(3).copied().unwrap_or(0);
    let optlen_ptr = args.get(4).copied().unwrap_or(0);
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    let mut lenbuf = [0u8; 4];
    if optlen_ptr == 0 || !ctx.mem.read(optlen_ptr, &mut lenbuf) {
        return MINUS_ONE;
    }
    let optlen = i32::from_le_bytes(lenbuf).clamp(0, 128) as usize;
    if optlen != 0 && (optval == 0 || !ctx.mem.write(optval, &vec![0u8; optlen])) {
        return MINUS_ONE;
    }
    if !ctx.mem.write(optlen_ptr, &(optlen as i32).to_le_bytes()) {
        return MINUS_ONE;
    }
    OK
}

/// `select(nfds, readfds, writefds, exceptfds, timeout)`: honor the timeout
/// (bounded like `nanosleep`), then report **zero descriptors ready** with
/// all supplied fd sets cleared — the honest offline answer (nothing readable,
/// nothing exceptional; write-readiness is deliberately not faked so a title
/// polls instead of streaming into a void).
fn hle_select(ctx: &HleContext, args: &[u64]) -> u64 {
    let nfds = args.first().copied().unwrap_or(0);
    let timeout_ptr = args.get(4).copied().unwrap_or(0);

    // Clear the words covering nfds descriptors in each non-null set.
    let set_bytes = (nfds.min(1024).div_ceil(8) as usize).next_multiple_of(8);
    for set_ptr in [args.get(1), args.get(2), args.get(3)] {
        let ptr = set_ptr.copied().unwrap_or(0);
        if ptr != 0 && set_bytes != 0 && !ctx.mem.write(ptr, &vec![0u8; set_bytes]) {
            return MINUS_ONE;
        }
    }

    // Sleep out (a bounded slice of) the caller's timeout so a select-poll
    // loop does not become a busy spin.
    const MAX_SLEEP_MS: u64 = 100;
    let mut sleep_ms = MAX_SLEEP_MS;
    if timeout_ptr != 0 {
        let mut tv = [0u8; 16];
        if ctx.mem.read(timeout_ptr, &mut tv) {
            let secs = i64::from_le_bytes(tv[..8].try_into().expect("fixed slice")).max(0) as u64;
            let usecs = i64::from_le_bytes(tv[8..].try_into().expect("fixed slice")).max(0) as u64;
            sleep_ms = secs
                .saturating_mul(1000)
                .saturating_add(usecs / 1000)
                .min(MAX_SLEEP_MS);
        }
    }
    if sleep_ms > 0 && !ctx.guest_threads.process_is_terminating() {
        ctx.services
            .sleep(std::time::Duration::from_millis(sleep_ms));
    }
    debug!("select(nfds={nfds}) -> 0 ready (offline; slept {sleep_ms}ms)");
    0
}

/// `socket(domain, type, protocol)`: hand back a fresh offline socket fd.
fn hle_socket(ctx: &HleContext, _args: &[u64]) -> u64 {
    let Some(fd) = ctx.services.create_socket() else {
        crate::libkernel::set_guest_errno(ctx, EMFILE);
        return MINUS_ONE;
    };
    debug!("socket() -> offline fd {fd:#x}");
    fd as u32 as u64
}

/// Parse a guest `sockaddr_in` (family byte at +1 == AF_INET, big-endian port
/// at +2, 4 IPv4 bytes at +4). Returns `(ip_bytes, port)`.
fn parse_sockaddr_in(ctx: &HleContext, addr: u64, addrlen: i64) -> Option<([u8; 4], u16)> {
    if addr == 0 || addrlen < 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(addr, &mut buf) {
        return None;
    }
    // Orbis uses the BSD `sin_len, sin_family` prefix. Some compiler/runtime
    // code still constructs the POSIX `sa_family_t` form (`02 00`) directly,
    // so accept both representations of AF_INET.
    if buf[1] != AF_INET as u8 && u16::from_le_bytes([buf[0], buf[1]]) != AF_INET as u16 {
        return None;
    }
    let port = u16::from_be_bytes([buf[2], buf[3]]);
    let ip = [buf[4], buf[5], buf[6], buf[7]];
    Some((ip, port))
}

/// `bind(fd, sockaddr, addrlen)`: record the bound address (offline).
fn hle_bind(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let sockaddr = args.get(1).copied().unwrap_or(0);
    let addrlen = args.get(2).copied().unwrap_or(0) as i64;
    let Some(mut sock) = ctx.kernel.kernel_sockets.get_mut(&fd) else {
        return MINUS_ONE;
    };
    let Some((ip, port)) = parse_sockaddr_in(ctx, sockaddr, addrlen) else {
        return MINUS_ONE;
    };
    sock.bound_ip = ip;
    sock.bound_port = port;
    sock.bound = true;
    OK
}

/// `connect(fd, sockaddr, addrlen)`: always fails — no host connectivity.
fn hle_connect(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    if !ctx.services.socket_exists(fd) {
        return MINUS_ONE;
    }
    debug!("connect(fd={fd:#x}) -> refused (offline)");
    MINUS_ONE
}

/// `setsockopt(fd, level, option, value, length)`: validate the guest option
/// payload and accept it for an offline socket. No option can enable host
/// connectivity; the observable socket behavior remains deterministic.
fn hle_setsockopt(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let value = args.get(3).copied().unwrap_or(0);
    let length = args.get(4).copied().unwrap_or(0);
    if !ctx.services.socket_exists(fd) || length > 4096 {
        return MINUS_ONE;
    }
    if length != 0 {
        let Ok(length) = usize::try_from(length) else {
            return MINUS_ONE;
        };
        let mut payload = vec![0u8; length];
        if value == 0 || !ctx.mem.read(value, &mut payload) {
            return MINUS_ONE;
        }
    }
    OK
}

/// sceNet-facing views of the offline socket operations above. The `sceNet*`
/// spellings in `libsce_net` return Orbis `SCE_NET_ERROR_*` codes instead of
/// `-1` + errno, so they wrap these rather than the POSIX handlers directly.
pub(crate) fn bind_offline(ctx: &HleContext, args: &[u64]) -> bool {
    hle_bind(ctx, args) == OK
}

/// `Some(bytes_sent)` on success, `None` when the payload was unreadable.
pub(crate) fn send_offline(ctx: &HleContext, args: &[u64]) -> Option<u64> {
    match hle_send(ctx, args) {
        MINUS_ONE => None,
        sent => Some(sent),
    }
}

pub(crate) fn setsockopt_offline(ctx: &HleContext, args: &[u64]) -> bool {
    hle_setsockopt(ctx, args) == OK
}

/// `getsockname(fd, sockaddr, addrlen)`: write back the bound `sockaddr_in`.
fn hle_getsockname(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let sockaddr = args.get(1).copied().unwrap_or(0);
    let addrlen_ptr = args.get(2).copied().unwrap_or(0);
    let Some(sock) = ctx.kernel.kernel_sockets.get(&fd) else {
        return MINUS_ONE;
    };
    if !sock.bound {
        return MINUS_ONE;
    }
    let mut lenbuf = [0u8; 4];
    if addrlen_ptr == 0 || !ctx.mem.read(addrlen_ptr, &mut lenbuf) {
        return MINUS_ONE;
    }
    let addrlen = i32::from_le_bytes(lenbuf);
    if addrlen < 8 {
        return MINUS_ONE;
    }
    // sockaddr_in: len(1)=16, family(1)=AF_INET, port(be16), ip(4), zero pad.
    let mut sa = [0u8; 16];
    sa[0] = 16;
    sa[1] = AF_INET as u8;
    sa[2..4].copy_from_slice(&sock.bound_port.to_be_bytes());
    sa[4..8].copy_from_slice(&sock.bound_ip);
    let write_len = (addrlen as usize).min(16);
    if !ctx.mem.write(sockaddr, &sa[..write_len]) {
        return MINUS_ONE;
    }
    if !ctx
        .mem
        .write(addrlen_ptr, &(write_len as i32).to_le_bytes())
    {
        return MINUS_ONE;
    }
    OK
}

/// `bzero(dst, len)`: zero `len` bytes of guest memory.
fn hle_bzero(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    if len > 0 && dst != 0 && !crate::zero_guest_range(ctx.mem, dst, len) {
        return MINUS_ONE;
    }
    OK
}

/// `inet_pton(af, src, dst)`: parse a dotted-quad IPv4 string into 4 bytes.
/// Returns 1 on success, 0 on a malformed address, -1 for a bad argument.
fn hle_inet_pton(ctx: &HleContext, args: &[u64]) -> u64 {
    let af = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let dst = args.get(2).copied().unwrap_or(0);
    if af != AF_INET || src == 0 || dst == 0 {
        return MINUS_ONE;
    }
    // Read the source string (bounded).
    let mut buf = [0u8; 64];
    if !ctx.mem.read(src, &mut buf) {
        return 0;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let Ok(text) = std::str::from_utf8(&buf[..end]) else {
        return 0;
    };
    let octets: Vec<&str> = text.split('.').filter(|s| !s.is_empty()).collect();
    if octets.len() != 4 {
        return 0;
    }
    let mut packed = [0u8; 4];
    for (i, part) in octets.iter().enumerate() {
        match part.parse::<u8>() {
            Ok(v) => packed[i] = v,
            Err(_) => return 0,
        }
    }
    if !ctx.mem.write(dst, &packed) {
        return 0;
    }
    1
}

/// `inet_ntop(AF_INET, src, dst, size)`: format four guest IPv4 bytes into a
/// bounded NUL-terminated dotted quad and return `dst` on success.
fn hle_inet_ntop(ctx: &HleContext, args: &[u64]) -> u64 {
    let af = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let dst = args.get(2).copied().unwrap_or(0);
    let size = args.get(3).copied().unwrap_or(0);
    if af != AF_INET || src == 0 || dst == 0 {
        return 0;
    }
    let mut ip = [0u8; 4];
    if !ctx.mem.read(src, &mut ip) {
        return 0;
    }
    let text = format!("{}.{}.{}.{}\0", ip[0], ip[1], ip[2], ip[3]);
    if size < text.len() as u64 || !ctx.mem.write(dst, text.as_bytes()) {
        return 0;
    }
    dst
}

/// `htons(value)`: host→network byte order for a 16-bit value.
fn hle_htons(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.first().copied().unwrap_or(0) as u16;
    u64::from(v.to_be())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    #[test]
    fn posix_provider_aliases_resolve_the_same_offline_impls() {
        // The measured Minecraft title imports socket/bind/connect/
        // getsockname/inet_pton naming libScePosix, not libkernel.
        let registry = HleRegistry::new();
        for name in ["socket", "bind", "connect", "getsockname", "inet_pton"] {
            assert!(
                registry.is_implemented("libScePosix", name),
                "libScePosix::{name} must be registered"
            );
        }
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let fd = registry
            .call(&ctx, "libScePosix", "socket", &[2, 1, 0])
            .expect("libScePosix::socket registered");
        assert!(fd < 0x8000_0000, "socket() must yield a small fd, not -1");
        assert_eq!(
            registry.call(&ctx, "libScePosix", "connect", &[fd, 0, 0]),
            Some(u64::MAX),
            "offline connect must refuse with -1"
        );
    }

    #[test]
    fn htons_swaps_bytes() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_htons(&ctx, &[0x1234]), 0x3412);
        assert_eq!(hle_htons(&ctx, &[80]), 0x5000); // port 80 → network order
    }

    #[test]
    fn inet_pton_parses_ipv4() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(ctx.mem.write(0x40, b"192.168.1.10\0"));
        assert_eq!(hle_inet_pton(&ctx, &[AF_INET, 0x40, 0x80]), 1);
        let mut b = [0u8; 4];
        assert!(ctx.mem.read(0x80, &mut b));
        assert_eq!(b, [192, 168, 1, 10]);
        // Malformed → 0.
        assert!(ctx.mem.write(0x40, b"not.an.ip\0"));
        assert_eq!(hle_inet_pton(&ctx, &[AF_INET, 0x40, 0x80]), 0);
        // Wrong family → -1.
        assert_eq!(hle_inet_pton(&ctx, &[10, 0x40, 0x80]), MINUS_ONE);
    }

    #[test]
    fn inet_ntop_formats_ipv4_and_checks_capacity() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(ctx.mem.write(0x40, &[192, 168, 1, 10]));
        assert_eq!(hle_inet_ntop(&ctx, &[AF_INET, 0x40, 0x80, 16]), 0x80);
        let mut text = [0u8; 13];
        assert!(ctx.mem.read(0x80, &mut text));
        assert_eq!(&text, b"192.168.1.10\0");
        assert_eq!(hle_inet_ntop(&ctx, &[AF_INET, 0x40, 0x90, 4]), 0);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "inet_ntop"));
    }

    #[test]
    fn bzero_clears_memory() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(ctx.mem.write(0x40, &[0xFFu8; 8]));
        assert_eq!(hle_bzero(&ctx, &[0x40, 8]), OK);
        let mut b = [0xAAu8; 8];
        assert!(ctx.mem.read(0x40, &mut b));
        assert_eq!(b, [0u8; 8]);
        assert_eq!(hle_bzero(&ctx, &[0x40, u64::MAX]), MINUS_ONE);
    }

    #[test]
    fn offline_posix_socket_surface_bounds_empty_receive_polling_and_never_connects() {
        use crate::GuestMemory;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // Nonzero base: the errno slot is arena-allocated, and address 0 would
        // read as "no errno slot".
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let fd = hle_socket(&ctx, &[]);

        // listen accepts; accept/recv report EWOULDBLOCK instead of blocking.
        assert_eq!(hle_listen(&ctx, &[fd, 8]), OK);
        assert_eq!(hle_accept(&ctx, &[fd, 0, 0]), MINUS_ONE);
        let recv_started = std::time::Instant::now();
        assert_eq!(hle_recv(&ctx, &[fd, 0x100, 16, 0]), MINUS_ONE);
        assert!(
            recv_started.elapsed() >= OFFLINE_RECV_BACKOFF,
            "an empty offline receive must yield instead of becoming a host-core spin"
        );
        let errno_slot = crate::libkernel::hle_error_addr(&ctx, &[]);
        let mut errno = [0u8; 4];
        assert!(mem.read(errno_slot, &mut errno));
        assert_eq!(i32::from_le_bytes(errno), EWOULDBLOCK);

        // send pretends success for a real buffer, faults a wild one.
        assert!(mem.write(0x100, b"PING"));
        assert_eq!(hle_send(&ctx, &[fd, 0x100, 4, 0]), 4);
        assert_eq!(hle_send(&ctx, &[fd, 0xFFFF_F000, 4, 0]), MINUS_ONE);

        // shutdown succeeds; getpeername writes a zeroed sockaddr.
        assert_eq!(hle_shutdown(&ctx, &[fd, 2]), OK);
        assert!(mem.write(0x200, &16i32.to_le_bytes()));
        assert_eq!(hle_getpeername(&ctx, &[fd, 0x208, 0x200]), OK);
        let mut sa = [0xFFu8; 8];
        assert!(mem.read(0x208, &mut sa));
        assert_eq!(sa[1], AF_INET as u8);
        assert_eq!(&sa[2..8], &[0u8; 6]);

        // getsockopt zero-fills the option value (e.g. SO_ERROR = no error).
        assert!(mem.write(0x240, &4i32.to_le_bytes()));
        assert!(mem.write(0x248, &[0xFFu8; 4]));
        assert_eq!(
            hle_getsockopt(&ctx, &[fd, 0xffff, 0x1007, 0x248, 0x240]),
            OK
        );
        let mut opt = [0xFFu8; 4];
        assert!(mem.read(0x248, &mut opt));
        assert_eq!(opt, [0u8; 4]);

        // select clears the supplied fd sets and reports zero ready.
        assert!(mem.write(0x300, &[0xFFu8; 16]));
        assert!(mem.write(0x310, &0i64.to_le_bytes())); // timeout 0s
        assert!(mem.write(0x318, &0i64.to_le_bytes())); // + 0us
        assert_eq!(hle_select(&ctx, &[64, 0x300, 0, 0, 0x310]), 0);
        let mut set = [0xFFu8; 8];
        assert!(mem.read(0x300, &mut set));
        assert_eq!(set, [0u8; 8]);

        // Every call on an unknown fd is -1.
        for result in [
            hle_listen(&ctx, &[0x999, 8]),
            hle_accept(&ctx, &[0x999, 0, 0]),
            hle_recv(&ctx, &[0x999, 0x100, 4, 0]),
            hle_send(&ctx, &[0x999, 0x100, 4, 0]),
            hle_shutdown(&ctx, &[0x999, 2]),
            hle_getpeername(&ctx, &[0x999, 0x208, 0x200]),
            hle_getsockopt(&ctx, &[0x999, 0, 0, 0x248, 0x240]),
        ] {
            assert_eq!(result, MINUS_ONE);
        }
    }

    #[test]
    fn socket_bind_getsockname_roundtrip_and_connect_fails() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let fd = hle_socket(&ctx, &[]);
        assert!(fd != MINUS_ONE && fd != 0);
        // Build a sockaddr_in for 10.0.0.5:8080 at 0x100.
        let mut sa = [0u8; 8];
        sa[1] = AF_INET as u8;
        sa[2..4].copy_from_slice(&8080u16.to_be_bytes());
        sa[4..8].copy_from_slice(&[10, 0, 0, 5]);
        assert!(ctx.mem.write(0x100, &sa));
        assert_eq!(hle_bind(&ctx, &[fd, 0x100, 8]), OK);
        // connect always refuses (offline).
        assert_eq!(hle_connect(&ctx, &[fd, 0x100, 8]), MINUS_ONE);
        // getsockname reads the bound address back.
        assert!(ctx.mem.write(0x200, &16i32.to_le_bytes())); // addrlen in
        assert_eq!(hle_getsockname(&ctx, &[fd, 0x208, 0x200]), OK);
        let mut out = [0u8; 8];
        assert!(ctx.mem.read(0x208, &mut out));
        assert_eq!(out[1], AF_INET as u8);
        assert_eq!(u16::from_be_bytes([out[2], out[3]]), 8080);
        assert_eq!(&out[4..8], &[10, 0, 0, 5]);
        // Operations on an unknown fd → -1.
        assert_eq!(hle_bind(&ctx, &[0x999, 0x100, 8]), MINUS_ONE);

        // POSIX `sa_family_t` prefix is also accepted.
        let mut posix_sa = sa;
        posix_sa[0..2].copy_from_slice(&(AF_INET as u16).to_le_bytes());
        assert!(ctx.mem.write(0x100, &posix_sa));
        assert_eq!(hle_bind(&ctx, &[fd, 0x100, 8]), OK);
        assert!(ctx.mem.write(0x240, &1u32.to_le_bytes()));
        assert_eq!(hle_setsockopt(&ctx, &[fd, 0xffff, 1, 0x240, 4]), OK);
        assert_eq!(hle_setsockopt(&ctx, &[0x999, 0, 0, 0, 0]), MINUS_ONE);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "setsockopt"));
    }
}
