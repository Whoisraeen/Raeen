//! HLE libSceVoice — voice chat capture/playback ports.
//!
//! **Offline semantics (Tier B, 2026-07-27):** Raeen models a console with the
//! voice library working but **no microphone and no chat session** — the
//! library initializes, ports are real handles that can be created, connected,
//! and deleted, but an output port never produces data (a read reports zero
//! bytes — silence) and written input data is accepted and dropped. This keeps
//! a title's voice bring-up path (GTA V imports 15 of these) alive without
//! fabricating a device.
//!
//! Cross-checked against shadPS4's GPL-2.0 `voice.cpp`/`voice.h` (its
//! `OrbisVoicePortInfo` layout and its `frame_size = 1` convention that avoids
//! divide-by-zero in callers); the port-registry behavior is Raeen's own —
//! shadPS4 stubs every entry point to `ORBIS_OK`. libSceVoice error codes are
//! not publicly documented and are NOT guessed: argument problems reuse the
//! generic kernel `EFAULT`/`EINVAL` spellings used elsewhere in this crate,
//! with a comment marking the uncertainty.

use crate::{HleContext, HleRegistry};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use tracing::debug;

const OK: u64 = 0;
/// Generic kernel EFAULT — used for bad out-pointers because the real
/// `SCE_VOICE_ERROR_*` values are not publicly documented (uncertain code).
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
/// Generic kernel EINVAL — same uncertainty note as above.
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;

/// Whether `sceVoiceInit` ran (diagnostic only; calls are tolerated either way
/// — failing hard on ordering would be a guess about undocumented behavior).
static VOICE_INITIALIZED: AtomicBool = AtomicBool::new(false);
/// Live port ids -> port type recorded at creation (0 when unreadable).
static PORTS: Mutex<Option<HashMap<u32, u32>>> = Mutex::new(None);
/// Monotonic port-id source (nonzero).
static NEXT_PORT_ID: AtomicU32 = AtomicU32::new(1);

/// `OrbisVoicePortInfo` (shadPS4 `voice.h`): `{ s32 port_type; s32 state;
/// u32* edge; u32 byte_count; u32 frame_size; u16 edge_count; u16 reserved; }`
/// — 28 bytes of fields (the trailing u64 alignment pad is left untouched).
const PORT_INFO_BYTES: usize = 28;
/// Offset of `frame_size` within `OrbisVoicePortInfo`.
const PORT_INFO_FRAME_SIZE_OFFSET: usize = 20;

/// Register the libSceVoice functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceVoice", "sceVoiceInit", hle_init);
    registry.register("libSceVoice", "sceVoiceEnd", hle_end);
    // Start/Stop: return-code-only service toggles. In the documented no-mic /
    // no-session model the running service still carries only silence, so
    // starting it changes nothing a caller can read — reads report zero bytes
    // either way, which is the honest offline state, not a skipped output.
    registry.register("libSceVoice", "sceVoiceStart", hle_ok);
    registry.register("libSceVoice", "sceVoiceStop", hle_ok);
    registry.register("libSceVoice", "sceVoiceCreatePort", hle_create_port);
    registry.register("libSceVoice", "sceVoiceDeletePort", hle_delete_port);
    registry.register(
        "libSceVoice",
        "sceVoiceConnectIPortToOPort",
        hle_connect_ports,
    );
    registry.register(
        "libSceVoice",
        "sceVoiceDisconnectIPortFromOPort",
        hle_connect_ports,
    );
    registry.register("libSceVoice", "sceVoiceWriteToIPort", hle_write_to_iport);
    registry.register("libSceVoice", "sceVoiceReadFromOPort", hle_read_from_oport);
    registry.register("libSceVoice", "sceVoiceGetPortInfo", hle_get_port_info);
    registry.register("libSceVoice", "sceVoiceGetBitRate", hle_get_bit_rate);
    registry.register("libSceVoice", "sceVoiceGetPortAttr", hle_get_port_attr);
    // Volume of silence is silence, and there are no worker threads to
    // configure — both setters are complete in the no-mic model.
    registry.register("libSceVoice", "sceVoiceSetVolume", hle_ok);
    registry.register("libSceVoice", "sceVoiceSetThreadsParams", hle_ok);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

/// `sceVoiceInit(pArg, version)`: mark the library initialized. Idempotent —
/// re-init is tolerated rather than guessing an undocumented already-init code.
fn hle_init(_ctx: &HleContext, _args: &[u64]) -> u64 {
    VOICE_INITIALIZED.store(true, Ordering::Relaxed);
    let mut ports = PORTS.lock().unwrap();
    if ports.is_none() {
        *ports = Some(HashMap::new());
    }
    debug!("sceVoiceInit() -> OK (no microphone: ports will carry silence)");
    OK
}

/// `sceVoiceEnd()`: tear down — all ports are released.
fn hle_end(_ctx: &HleContext, _args: &[u64]) -> u64 {
    VOICE_INITIALIZED.store(false, Ordering::Relaxed);
    *PORTS.lock().unwrap() = None;
    debug!("sceVoiceEnd()");
    OK
}

/// `sceVoiceCreatePort(u32 *portId, const OrbisVoicePortParam *param)`: hand
/// out a real, tracked port id. The port type (leading `s32` of the param) is
/// recorded for `GetPortInfo`; an unreadable param is tolerated as type 0.
fn hle_create_port(ctx: &HleContext, args: &[u64]) -> u64 {
    let port_out = args.first().copied().unwrap_or(0);
    let param = args.get(1).copied().unwrap_or(0);
    if port_out == 0 {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let mut port_type_bytes = [0u8; 4];
    let port_type = if param != 0 && ctx.mem.read(param, &mut port_type_bytes) {
        u32::from_le_bytes(port_type_bytes)
    } else {
        0
    };
    let id = NEXT_PORT_ID.fetch_add(1, Ordering::Relaxed);
    if !ctx.mem.write(port_out, &id.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    PORTS
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(id, port_type);
    debug!("sceVoiceCreatePort(type={port_type}) -> port {id}");
    OK
}

/// `sceVoiceDeletePort(portId)`: release the port. An unknown id is an
/// argument error (generic code — the real one is undocumented).
fn hle_delete_port(_ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as u32;
    match PORTS.lock().unwrap().as_mut().and_then(|p| p.remove(&id)) {
        Some(_) => OK,
        None => {
            debug!("sceVoiceDeletePort({id}) -> unknown port");
            SCE_ERROR_INVALID_ARGUMENT
        }
    }
}

/// `sceVoiceConnectIPortToOPort(ips, ops)` / `Disconnect...`: accept the
/// (dis)connection — routing is bookkeeping-free because no data ever flows.
fn hle_connect_ports(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceVoice(Dis)connect(ips={}, ops={}) -> OK (no data flows offline)",
        args.first().copied().unwrap_or(0) as u32,
        args.get(1).copied().unwrap_or(0) as u32
    );
    OK
}

/// `sceVoiceWriteToIPort(ips, data, u32 *size, frameGaps)`: accept and drop
/// the audio — `*size` is left as written by the caller (everything consumed).
fn hle_write_to_iport(_ctx: &HleContext, args: &[u64]) -> u64 {
    let data = args.get(1).copied().unwrap_or(0);
    if data == 0 {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceVoiceReadFromOPort(ops, data, u32 *size)`: **no voice data exists** —
/// report zero bytes read. Writing `*size = 0` is the honest "silence" answer;
/// the caller's buffer is never touched.
fn hle_read_from_oport(ctx: &HleContext, args: &[u64]) -> u64 {
    let size_out = args.get(2).copied().unwrap_or(0);
    if size_out != 0 && !ctx.mem.write(size_out, &0u32.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceVoiceGetPortInfo(portId, OrbisVoicePortInfo *info)`: report an idle
/// port — zero state/bytes/edges with the recorded port type and shadPS4's
/// `frame_size = 1` convention (a zero frame size divides callers by zero).
fn hle_get_port_info(ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as u32;
    let info = args.get(1).copied().unwrap_or(0);
    if info == 0 {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let Some(port_type) = PORTS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|p| p.get(&id).copied())
    else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    let mut bytes = [0u8; PORT_INFO_BYTES];
    bytes[0..4].copy_from_slice(&port_type.to_le_bytes());
    bytes[PORT_INFO_FRAME_SIZE_OFFSET..PORT_INFO_FRAME_SIZE_OFFSET + 4]
        .copy_from_slice(&1u32.to_le_bytes());
    if !ctx.mem.write(info, &bytes) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceVoiceGetBitRate(portId, u32 *bitrate)`: report the same nominal value
/// shadPS4 reports (48000) — cross-checked, not measured.
fn hle_get_bit_rate(ctx: &HleContext, args: &[u64]) -> u64 {
    let bitrate = args.get(1).copied().unwrap_or(0);
    if bitrate == 0 || !ctx.mem.write(bitrate, &48000u32.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceVoiceGetPortAttr(portId, attr, void *val, u32 valSize)`: attribute
/// meanings are undocumented; report a zeroed value (bounded to 8 bytes) so
/// the caller never reads uninitialized memory from a "successful" query.
fn hle_get_port_attr(ctx: &HleContext, args: &[u64]) -> u64 {
    let val = args.get(2).copied().unwrap_or(0);
    let val_size = args.get(3).copied().unwrap_or(0);
    if val == 0 {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let write_len = val_size.min(8) as usize;
    if write_len > 0 && !ctx.mem.write(val, &[0u8; 8][..write_len]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x1000),
            crate::TestAllocator::new(0),
        )
    }

    /// Full offline lifecycle: init OK, a port is a real tracked handle,
    /// output reads report silence (zero bytes), and teardown releases ports.
    #[test]
    fn voice_lifecycle_is_silent_but_real() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_init(&ctx, &[0, 4]), OK);

        // Port param at 0x200 with type 2 (OPort-ish); id written to 0x100.
        assert!(mem.write(0x200, &2u32.to_le_bytes()));
        assert_eq!(hle_create_port(&ctx, &[0x100, 0x200]), OK);
        let mut id_bytes = [0u8; 4];
        assert!(mem.read(0x100, &mut id_bytes));
        let id = u32::from_le_bytes(id_bytes);
        assert_ne!(id, 0, "a real port id was handed out");

        // GetPortInfo reports the recorded type, idle state, frame_size 1.
        assert!(mem.write(0x300, &[0xFFu8; PORT_INFO_BYTES]));
        assert_eq!(hle_get_port_info(&ctx, &[u64::from(id), 0x300]), OK);
        let mut info = [0u8; PORT_INFO_BYTES];
        assert!(mem.read(0x300, &mut info));
        assert_eq!(u32::from_le_bytes(info[0..4].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(info[20..24].try_into().unwrap()),
            1,
            "frame_size must be nonzero (caller divide-by-zero guard)"
        );

        // Reading an output port reports 0 bytes — silence, promptly.
        assert!(mem.write(0x400, &0xFFFF_FFFFu32.to_le_bytes()));
        assert_eq!(
            hle_read_from_oport(&ctx, &[u64::from(id), 0x500, 0x400]),
            OK
        );
        let mut size = [0u8; 4];
        assert!(mem.read(0x400, &mut size));
        assert_eq!(u32::from_le_bytes(size), 0, "no voice data offline");

        // Write is accepted (and dropped).
        assert_eq!(
            hle_write_to_iport(&ctx, &[u64::from(id), 0x600, 0x400, 0]),
            OK
        );

        // Delete releases; a second delete reports the unknown port.
        assert_eq!(hle_delete_port(&ctx, &[u64::from(id)]), OK);
        assert_eq!(
            hle_delete_port(&ctx, &[u64::from(id)]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_get_port_info(&ctx, &[u64::from(id), 0x300]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        assert_eq!(hle_end(&ctx, &[]), OK);
    }

    #[test]
    fn bitrate_and_attr_write_bounded_values() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_get_bit_rate(&ctx, &[1, 0x100]), OK);
        let mut b = [0u8; 4];
        assert!(mem.read(0x100, &mut b));
        assert_eq!(u32::from_le_bytes(b), 48000);
        assert_eq!(hle_get_bit_rate(&ctx, &[1, 0]), SCE_ERROR_MEMORY_FAULT);

        // Attr value is zeroed, bounded to 8 bytes even for huge valSize.
        assert!(mem.write(0x200, &[0xABu8; 16]));
        assert_eq!(hle_get_port_attr(&ctx, &[1, 0, 0x200, 1 << 30]), OK);
        let mut v = [0u8; 16];
        assert!(mem.read(0x200, &mut v));
        assert_eq!(&v[..8], &[0u8; 8]);
        assert_eq!(&v[8..], &[0xABu8; 8], "write is bounded to 8 bytes");
    }

    /// Every measured GTA V libSceVoice import resolves.
    #[test]
    fn measured_voice_imports_are_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceVoiceInit",
            "sceVoiceEnd",
            "sceVoiceStart",
            "sceVoiceStop",
            "sceVoiceCreatePort",
            "sceVoiceDeletePort",
            "sceVoiceConnectIPortToOPort",
            "sceVoiceDisconnectIPortFromOPort",
            "sceVoiceWriteToIPort",
            "sceVoiceReadFromOPort",
            "sceVoiceGetPortInfo",
            "sceVoiceGetBitRate",
            "sceVoiceGetPortAttr",
            "sceVoiceSetVolume",
            "sceVoiceSetThreadsParams",
        ] {
            assert!(
                registry.is_implemented("libSceVoice", name),
                "{name} must be registered"
            );
        }
    }
}
