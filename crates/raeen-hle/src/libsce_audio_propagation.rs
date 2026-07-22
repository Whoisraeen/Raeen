//! HLE libSceAudioPropagation — the acoustics ray-casting service (audio
//! occlusion/propagation paths through scene geometry).
//!
//! A faithful Rust port of SharpEmu's `AudioPropagationExports` (GPL-2.0),
//! which is deliberately fail-soft: `SystemQueryMemory` reports a modest
//! {size, alignment} working set so the caller's allocation succeeds, and
//! every other entry point simply succeeds — a title hears no occlusion, but
//! its audio init path completes instead of dying on an unresolved import.
//!
//! Measured: ASTRO.BOT faults on the `sceAudioPropagationSystemQueryMemory`
//! stub (NID 0xef1c80c6bbac2e4a) at "GAME: Resident Load end" (2026-07-21);
//! the remaining 21 NIDs come from the same boot missing-NID list and are
//! registered here with it. There is no HLE substitute for real pathing yet —
//! `SourceGetAudioPath(Count)` succeeding with no recorded paths means direct
//! sound only, which matches what an empty scene would produce.

use crate::{HleContext, HleRegistry};

const OK: u64 = 0;

/// Register the libSceAudioPropagation functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceAudioPropagation",
        "sceAudioPropagationSystemQueryMemory",
        hle_system_query_memory,
    );
    // Everything else is accept-and-succeed (SharpEmu: `=> ctx.SetReturn(Ok)`).
    for name in [
        "sceAudioPropagationSystemQueryInfo",
        "sceAudioPropagationSystemCreate",
        "sceAudioPropagationSystemDestroy",
        "sceAudioPropagationSystem",
        "sceAudioPropagationSystemMemoryInit",
        "sceAudioPropagationSystemOptionInit",
        "sceAudioPropagationSystemLock",
        "sceAudioPropagationSystemSetAttributes",
        "sceAudioPropagationSystemSetRays",
        "sceAudioPropagationSystemGetRays",
        "sceAudioPropagationSystemRegisterMaterial",
        "sceAudioPropagationSystemUnregisterMaterial",
        "sceAudioPropagationRoomCreate",
        "sceAudioPropagationRoomDestroy",
        "sceAudioPropagationPortalCreate",
        "sceAudioPropagationPortalDestroy",
        "sceAudioPropagationPortalSetAttributes",
        "sceAudioPropagationPortalSettingsInit",
        "sceAudioPropagationSourceCreate",
        "sceAudioPropagationSourceDestroy",
        "sceAudioPropagationSourceSetAttributes",
        "sceAudioPropagationSourceCalculateAudioPaths",
        "sceAudioPropagationSourceGetAudioPath",
        "sceAudioPropagationSourceGetAudioPathCount",
        "sceAudioPropagationSourceGetRays",
        "sceAudioPropagationSourceQueryInfo",
        "sceAudioPropagationSourceRender",
        "sceAudioPropagationSourceRenderInfoInit",
        "sceAudioPropagationSourceSetAudioPath",
        "sceAudioPropagationSourceSetAudioPaths",
    ] {
        registry.register("libSceAudioPropagation", name, |_, _| OK);
    }
}

/// `sceAudioPropagationSystemQueryMemory(params, outSizeAlign *)`: report the
/// working set `SystemCreate` expects the caller to provide. SharpEmu writes
/// {size = 1 MiB, alignment = 256 B} — modest and aligned so the caller's
/// allocation succeeds — and ignores a null out-pointer.
fn hle_system_query_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.get(1).copied().unwrap_or(0);
    if out != 0 {
        ctx.mem.write(out, &0x10_0000u64.to_le_bytes());
        ctx.mem.write(out + 8, &0x100u64.to_le_bytes());
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn query_memory_reports_a_modern_aligned_working_set() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_system_query_memory(&ctx, &[0, 0x20]), OK);
        let mut b = [0u8; 16];
        assert!(mem.read(0x20, &mut b));
        assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 0x10_0000);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 0x100);
        // A null out-pointer is tolerated.
        assert_eq!(hle_system_query_memory(&ctx, &[0, 0]), OK);
    }

    #[test]
    fn full_measured_surface_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceAudioPropagationSystemQueryMemory",
            "sceAudioPropagationSystemCreate",
            "sceAudioPropagationSystemDestroy",
            "sceAudioPropagationSystemSetAttributes",
            "sceAudioPropagationSystemSetRays",
            "sceAudioPropagationSystemGetRays",
            "sceAudioPropagationSystemRegisterMaterial",
            "sceAudioPropagationSystemUnregisterMaterial",
            "sceAudioPropagationRoomCreate",
            "sceAudioPropagationRoomDestroy",
            "sceAudioPropagationPortalCreate",
            "sceAudioPropagationPortalDestroy",
            "sceAudioPropagationPortalSetAttributes",
            "sceAudioPropagationSourceCreate",
            "sceAudioPropagationSourceDestroy",
            "sceAudioPropagationSourceSetAttributes",
            "sceAudioPropagationSourceCalculateAudioPaths",
            "sceAudioPropagationSourceGetAudioPath",
            "sceAudioPropagationSourceGetAudioPathCount",
            "sceAudioPropagationSourceGetRays",
            "sceAudioPropagationSourceRender",
            "sceAudioPropagationSourceSetAudioPaths",
        ] {
            assert!(
                registry.is_implemented("libSceAudioPropagation", name),
                "{name} must be registered"
            );
        }
    }
}
