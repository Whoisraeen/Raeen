//! HLE libkernel **BSD sockets** (offline) + pure network helpers.
//!
//! Ported from SharpEmu's `KernelSocketCompatExports` (GPL-2.0), **minus its
//! real host-TCP path**: XPS5X models no host connectivity (consistent with
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
    registry.register("libkernel", "bzero", hle_bzero);
    registry.register("libkernel", "inet_pton", hle_inet_pton);
    registry.register("libkernel", "htons", hle_htons);
}

/// `socket(domain, type, protocol)`: hand back a fresh offline socket fd.
fn hle_socket(ctx: &HleContext, _args: &[u64]) -> u64 {
    let fd = ctx.kernel.create_socket();
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
    if buf[1] != AF_INET as u8 {
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
    if !ctx.kernel.kernel_sockets.contains_key(&fd) {
        return MINUS_ONE;
    }
    debug!("connect(fd={fd:#x}) -> refused (offline)");
    MINUS_ONE
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
    let len = args.get(1).copied().unwrap_or(0) as usize;
    if len > 0 && dst != 0 {
        let zeros = vec![0u8; len];
        if !ctx.mem.write(dst, &zeros) {
            return MINUS_ONE;
        }
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
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
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
    fn bzero_clears_memory() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(ctx.mem.write(0x40, &[0xFFu8; 8]));
        assert_eq!(hle_bzero(&ctx, &[0x40, 8]), OK);
        let mut b = [0xAAu8; 8];
        assert!(ctx.mem.read(0x40, &mut b));
        assert_eq!(b, [0u8; 8]);
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
    }
}
