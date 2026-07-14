//! HLE libSceAudioOut — audio output management.
//!
//! A title's audio thread loops: fill a buffer, call `sceAudioOutOutput`
//! (which on real hardware *blocks* for roughly one buffer's playback time
//! so the ring buffer paces itself), repeat. XPS5X does not yet play audio,
//! but this stub must (a) never hang that thread and (b) not let it spin at
//! 100% CPU. So `Output` acknowledges the buffer, sleeps ~one buffer period
//! (grain ÷ frequency, bounded), and returns the sample count — real pacing
//! without real playback (CLAUDE.md M3: "audio stub must not hang the
//! title"). Export set cross-checked against SharpEmu.

use crate::{HleContext, HleRegistry};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// Upper bound on how long one `Output` blocks — a wild grain/frequency
/// can't wedge the audio thread for more than this.
const OUTPUT_MAX_SLEEP: Duration = Duration::from_millis(100);

/// Per-port `(grain_samples, frequency_hz)`, recorded at `Open` and read by
/// `Output` to pace itself. Keyed by the port handle we hand back.
fn ports() -> &'static Mutex<HashMap<u32, (u32, u32)>> {
    static PORTS: OnceLock<Mutex<HashMap<u32, (u32, u32)>>> = OnceLock::new();
    PORTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Monotonic port-handle counter (handles start at 1; 0 is never a valid
/// port).
static NEXT_HANDLE: AtomicU32 = AtomicU32::new(1);

/// Register libSceAudioOut HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAudioOut", "sceAudioOutInit", hle_ok);
    registry.register("libSceAudioOut", "sceAudioOutOpen", hle_open);
    registry.register("libSceAudioOut", "sceAudioOutOutput", hle_output);
    registry.register("libSceAudioOut", "sceAudioOutClose", hle_close);
    registry.register("libSceAudioOut", "sceAudioOutSetVolume", hle_ok);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceAudioOutOpen(userId, type, index, length, freq, param)`: records the
/// port's grain (`length`, samples per `Output`) and `freq`, returns a new
/// positive port handle.
fn hle_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    let grain = args.get(3).copied().unwrap_or(256) as u32;
    let freq = args.get(4).copied().unwrap_or(48_000) as u32;
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    debug!("sceAudioOutOpen(grain={grain}, freq={freq}) -> handle {handle}");
    ports().lock().unwrap().insert(handle, (grain, freq));
    handle as u64
}

/// `sceAudioOutOutput(handle, ptr)`: acknowledge the submitted buffer, sleep
/// ~one buffer period (grain ÷ freq, capped) so the audio thread paces
/// instead of spinning, and return the sample count. No samples are actually
/// played yet.
fn hle_output(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let (grain, freq) = ports()
        .lock()
        .unwrap()
        .get(&handle)
        .copied()
        .unwrap_or((256, 48_000));
    if freq > 0 {
        let period = Duration::from_secs_f64(grain as f64 / freq as f64).min(OUTPUT_MAX_SLEEP);
        std::thread::sleep(period);
    }
    grain as u64
}

/// `sceAudioOutClose(handle)`: drop the port's recorded pacing state.
fn hle_close(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    debug!("sceAudioOutClose(handle={handle})");
    ports().lock().unwrap().remove(&handle);
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_bits() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn open_returns_distinct_positive_handles() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        let h1 = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
        let h2 = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
        assert!(
            h1 > 0 && h2 > 0 && h1 != h2,
            "handles must be positive and distinct"
        );
    }

    #[test]
    fn output_returns_grain_and_paces_without_hanging() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        let h = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
        let t0 = std::time::Instant::now();
        let ret = hle_output(&ctx, &[h, 0x1000]);
        assert_eq!(ret, 256, "returns the grain (samples submitted)");
        // 256/48000 ≈ 5.3ms; it slept *something* but nowhere near the cap.
        assert!(t0.elapsed() < OUTPUT_MAX_SLEEP * 2, "must not hang");
        assert_eq!(hle_close(&ctx, &[h]), SCE_OK);
    }

    #[test]
    fn output_on_unknown_handle_uses_defaults_and_returns() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        // Never opened: falls back to default grain, still returns promptly.
        assert_eq!(hle_output(&ctx, &[9999, 0x1000]), 256);
    }
}
