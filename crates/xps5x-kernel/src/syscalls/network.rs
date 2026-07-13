//! Network syscall handlers.
//!
//! Translates PS5 BSD socket operations to host networking.
//! Most PS5 games use libSceNet.sprx rather than raw syscalls.

use crate::OrbisKernel;
use tracing::debug;
use xps5x_core::error::KernelError;

/// Stub implementation for network syscalls.
///
/// Most networking goes through HLE'd libSceNet, not raw syscalls.
/// These stubs handle the rare cases where games use direct socket calls.
pub fn sys_socket(
    _kernel: &OrbisKernel,
    domain: i32,
    socket_type: i32,
    protocol: i32,
) -> Result<u64, KernelError> {
    debug!(
        "socket(domain={}, type={}, protocol={}) -> stubbed",
        domain, socket_type, protocol
    );
    // Return a fake fd for now.
    Ok(100)
}
