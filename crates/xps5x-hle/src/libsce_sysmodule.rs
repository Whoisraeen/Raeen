//! HLE libSceSysmodule — system-module loader.
//!
//! A title calls `sceSysmoduleLoadModule(id)` to bring a system module
//! (`SCE_SYSMODULE_*` — e.g. libSceNp, libSceAjm) into its address space.
//! In XPS5X every system module is HLE'd: its exports are already
//! NID-registered in the HLE registry, so "loading" one is a no-op that
//! succeeds and its functions resolve regardless. Real *user* `.prx` loading
//! from disk (a title's own split-out libraries) is a separate concern —
//! that needs the file-backed module-load path (see the ledger's "PRX load
//! chain" row), not this system-module API.

use crate::{HleContext, HleRegistry};
use tracing::{debug, info};

/// `SCE_OK`.
const SCE_OK: u64 = 0;

/// Register libSceSysmodule HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceSysmodule", "sceSysmoduleLoadModule", hle_load_module);
    registry.register(
        "libSceSysmodule",
        "sceSysmoduleLoadModuleInternalWithArg",
        hle_load_module,
    );
    registry.register(
        "libSceSysmodule",
        "sceSysmoduleUnloadModule",
        hle_unload_module,
    );
    registry.register("libSceSysmodule", "sceSysmoduleIsLoaded", hle_is_loaded);
    // `sceSysmoduleGetModuleInfoForUnwind(addr, flags, info)` is a thin
    // wrapper over the kernel's `sceKernelGetModuleInfoForUnwind` (shadPS4
    // `sysmodule.cpp:31` delegates exactly this way, minus a system-module
    // name-hiding step XPS5X has no need for) — same 304-byte info ABI, same
    // EINVAL/EFAULT/ESRCH error surface.
    registry.register(
        "libSceSysmodule",
        "sceSysmoduleGetModuleInfoForUnwind",
        crate::libkernel::hle_get_module_info_for_unwind,
    );
}

/// `sceSysmoduleLoadModule(id, ...)`: succeeds — the module's functions are
/// HLE-registered, nothing needs to be brought into memory.
fn hle_load_module(_ctx: &HleContext, args: &[u64]) -> u64 {
    let module_id = args.first().copied().unwrap_or(0) as u32;
    info!("sceSysmoduleLoadModule(id={module_id:#x}) -> OK (HLE-backed)");
    SCE_OK
}

fn hle_unload_module(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceSysmoduleUnloadModule(id={:#x})",
        args.first().copied().unwrap_or(0)
    );
    SCE_OK
}

/// `sceSysmoduleIsLoaded(id)`: reports every system module as loaded (`0`),
/// since they're all HLE-available.
fn hle_is_loaded(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceSysmoduleIsLoaded(id={:#x}) -> loaded",
        args.first().copied().unwrap_or(0)
    );
    SCE_OK // 0 = loaded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_sysmodule_calls_report_success_and_loaded() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_load_module(&ctx, &[0x8000_0001]), SCE_OK);
        assert_eq!(
            hle_load_module(&ctx, &[]),
            SCE_OK,
            "empty args must not panic"
        );
        assert_eq!(hle_unload_module(&ctx, &[0x8000_0001]), SCE_OK);
        assert_eq!(hle_is_loaded(&ctx, &[0x8000_0001]), SCE_OK);
    }
}
