//! HLE libSceAudioIn — audio **capture** (microphone / headset input).
//!
//! This library had **zero** registrations in the crate before, so every
//! `sceAudioIn*` import resolved to nothing. Measured on Blasphemous II
//! (PPSA13580) via `cargo xtask nids coverage`: 5 unresolved imports —
//! `sceAudioInOpen`, `sceAudioInAsyncOpen`, `sceAudioInClose`,
//! `sceAudioInInput`, `sceAudioInGetSilentState`.
//!
//! # Semantics: a real port that captures real silence
//!
//! Raeen has no capture device. The honest model is therefore **not** "fail
//! every call" and **not** "return zero and pretend": it is a *working* input
//! port that yields silence, plus an explicit "no device" report through the
//! API's own channel for exactly that.
//!
//! * `Open`/`AsyncOpen` allocate a real port and return a real positive handle
//!   (a title stores it and passes it back; a fabricated handle it cannot close
//!   would leak, and an error here can make a title abort voice chat setup
//!   entirely).
//! * `Input` **zero-fills** the guest's destination buffer — digital silence in
//!   the port's own PCM layout — and returns the sample count, then paces one
//!   grain period so a capture thread neither hangs nor spins at 100% CPU. This
//!   is the same pacing rule `libsce_audio_out.rs` uses for `Output`, and the
//!   same simulated block KytyPS5 applies in `Audio::AudioInInput`
//!   (`reference/kytyps5`, `src/libs/audio.cpp:564-582`).
//! * `GetSilentState` reports `DEVICE_NONE`, which is precisely how shadPS4
//!   answers when no microphone is available
//!   (`reference/shadps4`, `src/core/libraries/audio/audioin.cpp:250-252`). A
//!   title that checks this learns the truth from the API rather than from
//!   silent buffers it has to guess about.
//!
//! Error codes and the handle encoding are re-implemented from shadPS4's
//! `audioin_error.h` / `audioin.cpp` (GPL-2.0) — values, not code.

use crate::{HleContext, HleRegistry};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;

/// `ORBIS_AUDIO_IN_ERROR_INVALID_HANDLE` (shadPS4 `audioin_error.h`).
const SCE_AUDIO_IN_ERROR_INVALID_HANDLE: u64 = 0x8026_0101;
/// `ORBIS_AUDIO_IN_ERROR_INVALID_FREQ`.
const SCE_AUDIO_IN_ERROR_INVALID_FREQ: u64 = 0x8026_0103;
/// `ORBIS_AUDIO_IN_ERROR_INVALID_POINTER`.
const SCE_AUDIO_IN_ERROR_INVALID_POINTER: u64 = 0x8026_0105;
/// `ORBIS_AUDIO_IN_ERROR_INVALID_PARAM`.
const SCE_AUDIO_IN_ERROR_INVALID_PARAM: u64 = 0x8026_0106;
/// `ORBIS_AUDIO_IN_ERROR_PORT_FULL`.
const SCE_AUDIO_IN_ERROR_PORT_FULL: u64 = 0x8026_0107;
/// `ORBIS_AUDIO_IN_ERROR_NOT_OPENED`.
const SCE_AUDIO_IN_ERROR_NOT_OPENED: u64 = 0x8026_0109;

/// `ORBIS_AUDIO_IN_SILENT_STATE_DEVICE_NONE` — "no microphone exists or it is
/// unavailable" (shadPS4 `audioin.h:25`). The permanent state here.
const SILENT_STATE_DEVICE_NONE: u64 = 0x0000_0001;

/// Concurrent capture ports. Real hardware allows a handful; this only has to
/// be large enough that a title's voice-chat setup never sees `PORT_FULL`.
const MAX_IN_PORTS: usize = 8;

/// Handle tag bits, matching shadPS4's `(type << 16) | port_id | 0x30000000`
/// encoding so a title that inspects the handle sees a plausible one.
const HANDLE_TAG: u64 = 0x3000_0000;

/// Upper bound on how long one `Input` blocks. A wild grain/frequency must not
/// wedge the capture thread — same guard as `libsce_audio_out.rs`.
const INPUT_MAX_SLEEP: Duration = Duration::from_millis(100);

/// Cap on one `Input` buffer written into guest memory. Capture grains are
/// 128–2048 samples; stereo S16 at a generous grain stays far under this.
const MAX_INPUT_BYTES: usize = 1 << 20;

/// One open capture port.
#[derive(Clone, Copy)]
struct InPort {
    /// Samples per `Input` call (the port's grain).
    grain: u32,
    /// Sample rate in Hz, used for pacing.
    freq: u32,
    /// 1 for mono, 2 for stereo.
    channels: u32,
}

/// `port_id -> port`. Its own table rather than a kernel resource because
/// nothing else in the emulator consumes capture ports yet.
fn ports() -> &'static Mutex<HashMap<u32, InPort>> {
    static PORTS: OnceLock<Mutex<HashMap<u32, InPort>>> = OnceLock::new();
    PORTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register libSceAudioIn HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAudioIn", "sceAudioInInit", hle_init);
    registry.register("libSceAudioIn", "sceAudioInOpen", hle_open);
    // `AsyncOpen` and `HqOpen` differ from `Open` only in the device quality /
    // completion model a real driver applies; with no device behind either,
    // both allocate the same silent port (shadPS4 also routes `HqOpen`
    // straight into `sceAudioInOpen`).
    registry.register("libSceAudioIn", "sceAudioInAsyncOpen", hle_open);
    registry.register("libSceAudioIn", "sceAudioInHqOpen", hle_open);
    registry.register("libSceAudioIn", "sceAudioInClose", hle_close);
    registry.register("libSceAudioIn", "sceAudioInInput", hle_input);
    registry.register(
        "libSceAudioIn",
        "sceAudioInGetSilentState",
        hle_get_silent_state,
    );
}

/// `sceAudioInInit()`: nothing to initialize without a device.
fn hle_init(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceAudioInInit() -> OK (no capture device)");
    SCE_OK
}

/// Decode `sceAudioInOpen`'s `param` into a channel count.
///
/// `0` = S16 mono, `2` = S16 stereo (KytyPS5 `src/libs/audio.cpp:791-800`,
/// shadPS4 `OrbisAudioInParamFormat`). Anything else is a genuine parameter
/// error — guessing a layout would make `Input` write the wrong number of bytes
/// into the guest's buffer.
fn decode_channels(param: u64) -> Option<u32> {
    match param & 0xF {
        0 => Some(1),
        2 => Some(2),
        _ => None,
    }
}

/// `sceAudioInOpen(userId, type, index, len, freq, param)`: allocate a capture
/// port and return its handle (positive) or a negative error.
fn hle_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    let type_ = args.get(1).copied().unwrap_or(0);
    let grain = args.get(3).copied().unwrap_or(0) as u32;
    let freq = args.get(4).copied().unwrap_or(0) as u32;
    let param = args.get(5).copied().unwrap_or(0);

    let Some(channels) = decode_channels(param) else {
        warn!("sceAudioInOpen: unsupported format param {param:#x} — INVALID_PARAM");
        return SCE_AUDIO_IN_ERROR_INVALID_PARAM;
    };
    if freq == 0 {
        warn!("sceAudioInOpen: freq 0 — INVALID_FREQ");
        return SCE_AUDIO_IN_ERROR_INVALID_FREQ;
    }
    if grain == 0 {
        warn!("sceAudioInOpen: grain 0 — INVALID_PARAM");
        return SCE_AUDIO_IN_ERROR_INVALID_PARAM;
    }

    let mut ports = ports().lock().unwrap_or_else(PoisonError::into_inner);
    let Some(port_id) = (0..MAX_IN_PORTS as u32).find(|id| !ports.contains_key(id)) else {
        warn!("sceAudioInOpen: all {MAX_IN_PORTS} capture ports in use — PORT_FULL");
        return SCE_AUDIO_IN_ERROR_PORT_FULL;
    };
    ports.insert(
        port_id,
        InPort {
            grain,
            freq,
            channels,
        },
    );
    let handle = HANDLE_TAG | ((type_ & 0xFFFF) << 16) | u64::from(port_id);
    debug!(
        "sceAudioInOpen(type={type_}, grain={grain}, freq={freq}, ch={channels}) -> handle \
         {handle:#x} (silent: no capture device)"
    );
    handle
}

/// Recover the port id from a handle, rejecting anything not currently open.
fn port_of(handle: u64) -> Option<(u32, InPort)> {
    let port_id = (handle & 0xFFFF) as u32;
    let ports = ports().lock().unwrap_or_else(PoisonError::into_inner);
    ports.get(&port_id).map(|port| (port_id, *port))
}

/// `sceAudioInClose(handle)`: release the port.
fn hle_close(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let port_id = (handle & 0xFFFF) as u32;
    let mut ports = ports().lock().unwrap_or_else(PoisonError::into_inner);
    if ports.remove(&port_id).is_none() {
        warn!("sceAudioInClose({handle:#x}): port was not open — NOT_OPENED");
        return SCE_AUDIO_IN_ERROR_NOT_OPENED;
    }
    debug!("sceAudioInClose({handle:#x})");
    SCE_OK
}

/// `sceAudioInInput(handle, dest)`: capture one grain of **silence** into the
/// guest buffer and return the sample count.
///
/// Zero-filling is the truthful answer for a machine with no microphone: it is
/// what a real port with a muted device delivers. Returning a count without
/// touching the buffer would leave the title decoding whatever was previously
/// there as if it were audio.
fn hle_input(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let dest = args.get(1).copied().unwrap_or(0);
    let Some((_, port)) = port_of(handle) else {
        warn!("sceAudioInInput({handle:#x}): no such open port — INVALID_HANDLE");
        return SCE_AUDIO_IN_ERROR_INVALID_HANDLE;
    };
    if dest == 0 {
        warn!("sceAudioInInput({handle:#x}): null destination — INVALID_POINTER");
        return SCE_AUDIO_IN_ERROR_INVALID_POINTER;
    }
    // S16 samples: 2 bytes per channel per sample.
    let byte_len = port.grain as usize * port.channels as usize * 2;
    if byte_len > MAX_INPUT_BYTES {
        warn!(
            "sceAudioInInput({handle:#x}): {byte_len} bytes exceeds the {MAX_INPUT_BYTES}-byte cap \
             — INVALID_PARAM"
        );
        return SCE_AUDIO_IN_ERROR_INVALID_PARAM;
    }
    if !ctx.mem.write(dest, &vec![0u8; byte_len]) {
        warn!("sceAudioInInput({handle:#x}): dest {dest:#x} is not writable — INVALID_POINTER");
        return SCE_AUDIO_IN_ERROR_INVALID_POINTER;
    }
    // Pace one grain period so a capture thread runs at real time instead of
    // spinning. Bounded, like `libsce_audio_out.rs`'s `Output`.
    let period =
        Duration::from_secs_f64(f64::from(port.grain) / f64::from(port.freq)).min(INPUT_MAX_SLEEP);
    ctx.services.sleep(period);
    u64::from(port.grain)
}

/// `sceAudioInGetSilentState(handle)`: report why the input is silent.
///
/// `DEVICE_NONE` permanently — there is no capture device. This is the API's own
/// way of saying so, and reporting `0` ("not silent") would be a lie the title
/// acts on.
fn hle_get_silent_state(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    if port_of(handle).is_none() {
        warn!("sceAudioInGetSilentState({handle:#x}): no such open port — INVALID_HANDLE");
        return SCE_AUDIO_IN_ERROR_INVALID_HANDLE;
    }
    SILENT_STATE_DEVICE_NONE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GuestMemory;

    /// The port table is process-global (it models one console's capture
    /// hardware), so tests that clear it must not interleave.
    static PORT_TABLE_LOCK: Mutex<()> = Mutex::new(());

    fn ctx_bits(
        size: usize,
    ) -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(size),
            crate::TestAllocator::new(0),
        )
    }

    /// Open -> Input -> GetSilentState -> Close is a complete, honest cycle:
    /// a real handle, a zero-filled buffer, an explicit "no device" report,
    /// and a released port. No sleeps long enough to matter (grain/freq here is
    /// well under a millisecond), so the test is deterministic.
    #[test]
    fn capture_port_yields_silence_and_reports_no_device() {
        let _serialize = PORT_TABLE_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let (k, m, a) = ctx_bits(0x1000);
        let ctx = crate::test_ctx(&k, &m, &a);
        ports()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();

        // Poison the destination so zero-fill is observable rather than assumed.
        let dest = 0x400u64;
        assert!(m.write(dest, &[0xAB; 64]));

        // userId, type=1, index=0, grain=16, freq=16000, param=2 (S16 stereo).
        let handle = hle_open(&ctx, &[255, 1, 0, 16, 16_000, 2]);
        assert!(
            handle < 0x8000_0000,
            "open must return a positive handle, got {handle:#x}"
        );
        assert_eq!(handle & 0xFFFF_0000, HANDLE_TAG | (1 << 16));

        assert_eq!(hle_input(&ctx, &[handle, dest]), 16, "returns the grain");
        let mut got = [0xFFu8; 64];
        assert!(m.read(dest, &mut got));
        assert_eq!(got, [0u8; 64], "the guest buffer must be digital silence");

        assert_eq!(
            hle_get_silent_state(&ctx, &[handle]),
            SILENT_STATE_DEVICE_NONE
        );
        assert_eq!(hle_close(&ctx, &[handle]), SCE_OK);
        // A closed port is gone: the handle no longer captures or reports.
        assert_eq!(
            hle_input(&ctx, &[handle, dest]),
            SCE_AUDIO_IN_ERROR_INVALID_HANDLE
        );
        assert_eq!(hle_close(&ctx, &[handle]), SCE_AUDIO_IN_ERROR_NOT_OPENED);
    }

    /// Bad arguments produce the real error codes instead of a usable handle.
    #[test]
    fn open_rejects_unsupported_format_and_zero_rates() {
        let _serialize = PORT_TABLE_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let (k, m, a) = ctx_bits(0x100);
        let ctx = crate::test_ctx(&k, &m, &a);
        ports()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();

        // param 1 is not a documented AudioIn format.
        assert_eq!(
            hle_open(&ctx, &[255, 1, 0, 256, 48_000, 1]),
            SCE_AUDIO_IN_ERROR_INVALID_PARAM
        );
        assert_eq!(
            hle_open(&ctx, &[255, 1, 0, 256, 0, 2]),
            SCE_AUDIO_IN_ERROR_INVALID_FREQ
        );
        assert_eq!(
            hle_open(&ctx, &[255, 1, 0, 0, 48_000, 2]),
            SCE_AUDIO_IN_ERROR_INVALID_PARAM
        );
        assert!(
            ports()
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty(),
            "a rejected open must not consume a port"
        );
    }

    /// Every NID the measured title imports must be registered — the reason
    /// this module exists.
    #[test]
    fn every_measured_import_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceAudioInOpen",
            "sceAudioInAsyncOpen",
            "sceAudioInClose",
            "sceAudioInInput",
            "sceAudioInGetSilentState",
        ] {
            assert!(
                registry.is_implemented("libSceAudioIn", name),
                "libSceAudioIn::{name} is imported by the measured title"
            );
        }
    }
}
