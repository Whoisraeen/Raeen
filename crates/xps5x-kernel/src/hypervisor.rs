//! Hypervisor stub for the emulated PS5.
//!
//! The PS5 uses a hypervisor for Virtualization-Based Security (VBS),
//! NOT for running VMs. It enforces kernel integrity and code signing.
//! Games don't interact with the HV directly — it's transparent.
//!
//! XPS5X stubs out all hypervisor calls, effectively disabling the
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
            debug!("Hypercall {:#x} (args: {:?}) -> stubbed (success)", call_id, args);
        }
        0 // Success.
    }

    /// CPUID spoofing — report PS5-like CPU features.
    ///
    /// Some games check CPUID to detect hardware features.
    /// We return values matching the PS5's Zen 2 CPU.
    pub fn spoof_cpuid(&self, leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
        match leaf {
            // Vendor string: "AuthenticAMD"
            0x0000_0000 => (0x10, 0x6874_7541, 0x444D_4163, 0x6974_6E65),
            // Family/Model: Zen 2 (Family 17h, Model 60h = PS5 custom)
            0x0000_0001 => (0x00860F01, 0x0010_0800, 0x7ED8_320B, 0x178B_FBFF),
            // Extended features
            0x0000_0007 => {
                match subleaf {
                    0 => (0, 0x219C_97A9, 0x0000_0040, 0),
                    _ => (0, 0, 0, 0),
                }
            }
            _ => (0, 0, 0, 0),
        }
    }
}

impl Default for HypervisorStub {
    fn default() -> Self {
        Self::new()
    }
}
