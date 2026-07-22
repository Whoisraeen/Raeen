//! HLE libSceAcm — the PS5 Audio Codec Module (hardware-assisted audio
//! decode contexts and batches).
//!
//! No open reference implements this library (it is PS5-only; Kyty, SharpEmu
//! and shadPS4 are all silent on it), so every signature here is UNKNOWN and
//! these are honest unblock stubs, not a port: each call logs its arguments
//! the first time it is reached and reports success without writing any
//! guest memory. Batches "complete" instantly, which degrades to silent
//! audio — the measured title (Minecraft) calls `sceAcmContextCreate` on its
//! MAIN thread during boot, so an unresolved import here was the difference
//! between reaching the render loop and dying in audio init.
//!
//! When a real dump of the call arguments exists, replace the logged guesses
//! with the measured ABI.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::warn;

const OK: u64 = 0;

/// Register the libSceAcm functions the measured title imports.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAcm", "sceAcmContextCreate", hle_context_create);
    registry.register("libSceAcm", "sceAcmContextDestroy", |_, _| OK);
    registry.register(
        "libSceAcm",
        "sceAcmBatchStartBuffers",
        hle_batch_start_buffers,
    );
    registry.register("libSceAcm", "sceAcmBatchWait", |_, _| OK);
    registry.register("libSceAcm", "sceAcm_ConvReverb_SharedInput", |_, _| OK);
}

/// `sceAcmContextCreate(...)` — signature unknown. Logged once so a real run
/// records the argument shape; succeeds without writing guest memory (a
/// caller that reads an out-param handle gets whatever it pre-initialized,
/// which the paired all-accepting stubs above tolerate).
fn hle_context_create(_ctx: &HleContext, args: &[u64]) -> u64 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        warn!(
            ?args,
            "sceAcmContextCreate: UNKNOWN ABI — succeeding without side effects \
             (audio decode degrades to silence); record these args to reverse the signature"
        );
    }
    OK
}

/// `sceAcmBatchStartBuffers(...)` — signature unknown; the batch "completes"
/// before `sceAcmBatchWait` is ever called, so waits return instantly.
fn hle_batch_start_buffers(_ctx: &HleContext, args: &[u64]) -> u64 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        warn!(
            ?args,
            "sceAcmBatchStartBuffers: UNKNOWN ABI — reporting instant completion \
             (no samples are decoded; audio is silence)"
        );
    }
    OK
}
