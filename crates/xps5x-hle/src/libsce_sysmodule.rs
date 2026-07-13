//! HLE libSceSysmodule — Module loader.

use crate::{HleContext, HleRegistry};
use tracing::{debug, info};

/// Register libSceSysmodule HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceSysmodule", "sceSysmoduleLoadModule", hle_load_module);
    registry.register("libSceSysmodule", "sceSysmoduleUnloadModule", hle_unload_module);
    registry.register("libSceSysmodule", "sceSysmoduleIsLoaded", hle_is_loaded);
}

fn hle_load_module(_ctx: &HleContext, args: &[u64]) -> u64 {
    let module_id = args[0] as u32;
    info!("sceSysmoduleLoadModule(id={:#x})", module_id);
    // Always succeed — modules are internally HLE'd, not actually loaded.
    0
}

fn hle_unload_module(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("sceSysmoduleUnloadModule(id={:#x})", args[0]);
    0
}

fn hle_is_loaded(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("sceSysmoduleIsLoaded(id={:#x}) -> yes", args[0]);
    0 // 0 = loaded.
}
