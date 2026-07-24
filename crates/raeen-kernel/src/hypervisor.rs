//! Hypervisor stub for the emulated PS5.
//!
//! The PS5 uses a hypervisor for Virtualization-Based Security (VBS),
//! NOT for running VMs. It enforces kernel integrity and code signing.
//! Games don't interact with the HV directly — it's transparent.
//!
//! Raeen stubs out all hypervisor calls, effectively disabling the
//! security enforcement while maintaining API compatibility.

use tracing::debug;

/// Stub hypervisor state.
pub struct HypervisorStub {
    /// Whether to log HV calls (verbose debugging).
    verbose: bool,
}

impl HypervisorStub {
    pub fn new() -> Self {
        debug!("Hypervisor stub initialized (security enforcement disabled)");
        Self { verbose: false }
    }

    /// Handle a hypercall from the emulated kernel.
    ///
    /// All hypercalls are stubbed — we return success without
    /// performing any security checks.
    pub fn handle_hypercall(&self, call_id: u64, args: &[u64]) -> u64 {
        if self.verbose {
            debug!(
                "Hypercall {:#x} (args: {:?}) -> stubbed (success)",
                call_id, args
            );
        }
        0 // Success.
    }

    /// CPUID spoofing — report PS5-like CPU features.
    ///
    /// Some games check CPUID to detect hardware features.
    /// We return values matching the PS5's Zen 2 CPU.
    pub fn spoof_cpuid(&self, leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
        ps5_cpuid(leaf, subleaf)
    }
}

impl Default for HypervisorStub {
    fn default() -> Self {
        Self::new()
    }
}

/// CPUID values exposed to directly executing guest code.
///
/// Keep the feature set to instructions present on the PS5 Zen 2 rather than
/// leaking a newer host's family/model and optional ISA extensions. Leaves
/// whose layout is host-independent but needed for XSAVE use the host value;
/// the advertised Zen 2 feature mask remains the authority.
#[must_use]
pub fn ps5_cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    match leaf {
        // Maximum standard leaf and vendor string "AuthenticAMD".
        0x0000_0000 => (0x10, 0x6874_7541, 0x444D_4163, 0x6974_6E65),
        // Family 17h, model 60h, stepping 1; 8 cores / 16 logical processors.
        0x0000_0001 => (0x0086_0F01, 0x0010_0800, 0x7ED8_320B, 0x178B_FBFF),
        0x0000_0007 => match subleaf {
            // BMI1/2, AVX2, SMEP/SMAP, RDSEED, SHA and the Zen 2-era subset.
            0 => (0, 0x219C_97A9, 0x0000_0040, 0),
            _ => (0, 0, 0, 0),
        },
        // XSAVE component sizes/offsets are an ABI with the actual host OS.
        // Returning the host values is safe because leaf 1/7 above never
        // advertise a state component outside the PS5 feature set.
        0x0000_000D => host_cpuid(leaf, subleaf),
        0x8000_0000 => (0x8000_0008, 0, 0, 0),
        0x8000_0001 => (
            0x0086_0F01,
            0,
            0x0000_0121,
            0x2FD3_FBFF, // NX, 1GiB pages, RDTSCP, long mode.
        ),
        0x8000_0002 => brand_leaf(b"AMD Custom 8-Core Processor              ", 0),
        0x8000_0003 => brand_leaf(b"AMD Custom 8-Core Processor              ", 16),
        0x8000_0004 => brand_leaf(b"AMD Custom 8-Core Processor              ", 32),
        // Do not leak the host's advanced-power / invariant-TSC capabilities.
        // The console profile models this leaf as zero, which also keeps the
        // guest-visible CPU profile deterministic across host processors.
        0x8000_0007 => (0, 0, 0, 0),
        0x8000_0008 => (0x0000_3030, 0, 7, 0), // 48-bit physical/virtual, 8 cores.
        _ => (0, 0, 0, 0),
    }
}

#[cfg(target_arch = "x86_64")]
fn host_cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    // `__cpuid_count` is a safe intrinsic on x86-64 and does not access memory.
    let result = core::arch::x86_64::__cpuid_count(leaf, subleaf);
    (result.eax, result.ebx, result.ecx, result.edx)
}

#[cfg(not(target_arch = "x86_64"))]
fn host_cpuid(_leaf: u32, _subleaf: u32) -> (u32, u32, u32, u32) {
    (0, 0, 0, 0)
}

fn brand_leaf(brand: &[u8], offset: usize) -> (u32, u32, u32, u32) {
    let mut chunk = [0u8; 16];
    let available = brand.len().saturating_sub(offset).min(chunk.len());
    chunk[..available].copy_from_slice(&brand[offset..offset + available]);
    (
        u32::from_le_bytes(chunk[0..4].try_into().expect("brand eax")),
        u32::from_le_bytes(chunk[4..8].try_into().expect("brand ebx")),
        u32::from_le_bytes(chunk[8..12].try_into().expect("brand ecx")),
        u32::from_le_bytes(chunk[12..16].try_into().expect("brand edx")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpuid_reports_ps5_zen2_identity_and_topology() {
        let (_, ebx, ecx, edx) = ps5_cpuid(0, 0);
        let mut vendor = Vec::new();
        vendor.extend_from_slice(&ebx.to_le_bytes());
        vendor.extend_from_slice(&edx.to_le_bytes());
        vendor.extend_from_slice(&ecx.to_le_bytes());
        assert_eq!(&vendor, b"AuthenticAMD");

        let (eax, ebx, _, _) = ps5_cpuid(1, 0);
        assert_eq!((eax >> 20) & 0xff, 8);
        assert_eq!((eax >> 16) & 0xf, 6);
        assert_eq!((ebx >> 16) & 0xff, 16);
        assert_eq!(ps5_cpuid(0x8000_0008, 0).2 & 0xff, 7);
        assert_eq!(
            ps5_cpuid(0x8000_0007, 0).3 & (1 << 8),
            0,
            "do not leak the host invariant-TSC capability into the PS5 profile"
        );
    }
}
