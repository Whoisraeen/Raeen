//! HLE libScePad — Controller input interface.

use crate::{HleContext, HleRegistry};
use tracing::debug;

/// Register libScePad HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePad", "scePadInit", hle_pad_init);
    registry.register("libScePad", "scePadOpen", hle_pad_open);
    registry.register("libScePad", "scePadReadState", hle_pad_read_state);
    registry.register("libScePad", "scePadSetVibration", hle_pad_set_vibration);
}

fn hle_pad_init(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("scePadInit()");
    0
}

fn hle_pad_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePadOpen(userId={}, type={}, index={})", args[0], args[1], args[2]);
    1 // Return pad handle = 1.
}

fn hle_pad_read_state(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePadReadState(handle={}, data={:#x})", args[0], args[1]);
    0
}

fn hle_pad_set_vibration(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePadSetVibration(handle={}, ...)", args[0]);
    0
}
