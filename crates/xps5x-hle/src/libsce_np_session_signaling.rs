//! HLE libSceNpSessionSignaling — the P2P session-signaling init handshake.
//!
//! A faithful Rust port of SharpEmu's `NpSessionSignalingExports` (GPL-2.0).
//! Session signaling brokers peer-to-peer connectivity for PSN multiplayer
//! sessions. XPS5X has no PSN/networking backend, so the single exported
//! `Initialize` is an honest no-op that reports success — a title can complete
//! its signaling init and move on, but no peer connection is ever established.

use crate::HleRegistry;

const OK: u64 = 0;

/// Register the libSceNpSessionSignaling functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceNpSessionSignaling",
        "sceNpSessionSignalingInitialize",
        |_, _| OK,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    #[test]
    fn initialize_succeeds() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let reg = HleRegistry::new();
        assert_eq!(
            reg.call(
                &ctx,
                "libSceNpSessionSignaling",
                "sceNpSessionSignalingInitialize",
                &[]
            ),
            Some(OK)
        );
    }
}
