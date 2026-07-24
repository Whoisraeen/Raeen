//! HLE libSceAudioOut2 — the PS5 (Gen5) AudioOut2 context/port lifecycle **and**
//! real PCM playback.
//!
//! A Rust port of SharpEmu's `AudioOut2Exports` (GPL-2.0) for the lifecycle,
//! plus the actual output path (which SharpEmu's AudioOut2 never implemented —
//! its `ContextPush` only paces). `libSceAudioOut2` is the Gen5 audio-output API
//! (distinct from the older `libSceAudioOut`, ported in
//! [`crate::libsce_audio_out`]). GTA V and other AudioOut2 titles now reach the
//! host mixer: a port's PCM buffer is converted to interleaved-stereo f32 and
//! handed to [`raeen_audio::output::submit`].
//!
//! ## Buffer model (measured against KytyPS5 `libAudio2.cpp`, the live PS5 fork)
//!
//! The *real* AudioOut2 ABI carries **no** PCM pointer in `ContextPush` — its
//! prototype is `sceAudioOut2ContextPush(context, blocking)`. Instead:
//! - `PortCreate` records the port's **format** from its `AudioOut2PortParam`
//!   (`data_format` → channels + s16/float, `sampling_freq`), and associates the
//!   port with its owning context (for the grain / frame count).
//! - `PortSetAttributes` registers the port's **PCM buffer pointer** via the PCM
//!   attribute (`attribute_id == 0` → `AudioOut2Pcm { const void* data }`). This
//!   is KytyPS5's `AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM` path.
//! - `ContextPush` then submits every port of that context whose buffer is set:
//!   read `grain × channels × bytes_per_sample` bytes, convert, submit.
//!
//! The write path stays **fail-safe**: a port with no PCM buffer, a bad pointer,
//! a malformed frame/format, or a context with no ports just paces exactly as
//! before (no audio, never a crash). Because the SharpEmu-derived `PortCreate`
//! register convention differs from the real one on *which* argument holds the
//! context handle, the owning context is resolved by matching a live context
//! handle among the call's arguments — see [`hle_port_create`].
//!
//! Struct sizes/offsets for the lifecycle are ported from SharpEmu, whose
//! `AudioOut2ContextParamSize` (0x30) note records real guest evidence (Quake's
//! stack canary at param+0x60): only the populated prefix is written, well
//! below the canary. Handles/port ids are module-level monotonic counters
//! matching SharpEmu's statics. `OrbisGen2Result` codes map to the real Orbis
//! `EINVAL`/`EFAULT` (`0x8002_0016`/`0x8002_000E`) as plain zero-extended `u64`.

use crate::{GuestMemory, HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

const CONTEXT_PARAM_SIZE: usize = 0x30;
const CONTEXT_MEMORY_SIZE: u64 = 0x10000;

/// Upper bound on one port's PCM read from the guest, mirroring
/// [`crate::libsce_audio_out`]'s `MAX_OUTPUT_BYTES`. A grain of 0x4000 frames ×
/// 16 channels × f32 is exactly 1 MiB, so this admits the largest sane buffer
/// and rejects anything a malformed grain/format could inflate it to. Hardening
/// mirrored from SharpEmu `e13cb28` (bounds-check guest-buffer reads so a bad
/// frame count / format / pointer cannot overrun).
const MAX_PCM_BYTES: usize = 1 << 20;

/// Clamp on channel count when sizing a PCM read (KytyPS5 clamps `data_format`
/// channels to 16). Keeps `grain × channels × bps` from overflowing on a wild
/// `data_format`.
const MAX_CHANNELS: u32 = 16;

/// Clamp on the `AudioOut2PortSetAttributes` attribute count we will walk, so a
/// malformed `num` can't spin the parse loop. Real callers pass a handful.
const MAX_ATTRIBUTES: u64 = 64;

/// `AudioOut2Attribute` stride: `{ u32 id; i32 reserved; const void* value;
/// size_t value_size }` = 4 + 4 + 8 + 8 on the guest's LP64 ABI.
const ATTRIBUTE_STRIDE: u64 = 24;

/// KytyPS5 `AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM`: the attribute whose `value`
/// points at an `AudioOut2Pcm { const void* data }`.
const ATTRIBUTE_ID_PCM: u32 = 0;

// SharpEmu statics: `_nextContextHandle`/`_nextUserHandle` start at 1 (each
// incremented before first use, so first handle is 2); `_nextPortId` starts at
// 0 (first port id is 1).
static NEXT_CONTEXT_HANDLE: AtomicI64 = AtomicI64::new(1);
static NEXT_USER_HANDLE: AtomicI64 = AtomicI64::new(1);
static NEXT_PORT_ID: AtomicI32 = AtomicI32::new(0);

/// Register the libSceAudioOut2 functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAudioOut2", "sceAudioOut2Initialize", |_, _| OK);
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextResetParam",
        hle_context_reset_param,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextQueryMemory",
        hle_context_query_memory,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextCreate",
        hle_context_create,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextDestroy",
        hle_context_destroy,
    );
    // The streaming path the mixer thread drives once audio init completes.
    // Measured: Minecraft's FMOD output thread died jumping to
    // `sceAudioOut2PortSetAttributes` (NID 0xf174c0ad23f25879).
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextPush",
        hle_context_push,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextAdvance",
        hle_context_advance,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2PortSetAttributes",
        hle_port_set_attributes,
    );
    // Resolved at runtime via dlsym by the measured title's FMOD build, so it
    // never shows in the static import table — register it anyway.
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2ContextGetQueueLevel",
        hle_context_get_queue_level,
    );
    registry.register("libSceAudioOut2", "sceAudioOut2PortCreate", hle_port_create);
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2PortGetState",
        hle_port_get_state,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2GetSpeakerInfo",
        hle_get_speaker_info,
    );
    registry.register(
        "libSceAudioOut2",
        "sceAudioOut2PortDestroy",
        hle_port_destroy,
    );
    registry.register("libSceAudioOut2", "sceAudioOut2UserDestroy", |_, _| OK);
    registry.register("libSceAudioOut2", "sceAudioOut2UserCreate", hle_user_create);
}

/// `sceAudioOut2ContextResetParam(param)`: fills the 0x30-byte context param
/// with defaults (size, 2 channels, 48000 Hz, 0x400-frame grain).
fn hle_context_reset_param(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    if param == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; CONTEXT_PARAM_SIZE];
    buf[0x00..0x04].copy_from_slice(&(CONTEXT_PARAM_SIZE as u32).to_le_bytes());
    buf[0x04..0x08].copy_from_slice(&2u32.to_le_bytes());
    buf[0x08..0x0C].copy_from_slice(&48000u32.to_le_bytes());
    buf[0x0C..0x10].copy_from_slice(&0x400u32.to_le_bytes());
    if ctx.mem.write(param, &buf) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAudioOut2ContextQueryMemory(param, outMemorySize)`: reports the
/// context's required working-memory size (0x10000).
fn hle_context_query_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    let out_size = args.get(1).copied().unwrap_or(0);
    if param == 0 || out_size == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if ctx.mem.write(out_size, &CONTEXT_MEMORY_SIZE.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAudioOut2ContextCreate(param, memory, memorySize, outContext)`: returns
/// a fresh opaque context handle (u64) in `*outContext`, and records the
/// context's playback cadence (frequency + grain from the param block) so
/// `ContextPush`/`ContextAdvance` can pace the feeder to real hardware timing.
fn hle_context_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    let memory = args.get(1).copied().unwrap_or(0);
    let memory_size = args.get(2).copied().unwrap_or(0);
    let out_context = args.get(3).copied().unwrap_or(0);
    if param == 0 || memory == 0 || memory_size == 0 || out_context == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let handle = (NEXT_CONTEXT_HANDLE.fetch_add(1, Ordering::Relaxed) + 1) as u64;
    // param layout (CONTEXT_PARAM_SIZE block): +0x04 channels, +0x08
    // frequency, +0x0C grain samples — same offsets ContextResetParam fills.
    let mut pbuf = [0u8; CONTEXT_PARAM_SIZE];
    if ctx.mem.read(param, &mut pbuf) {
        let frequency = u32::from_le_bytes(pbuf[0x08..0x0C].try_into().expect("fixed slice"));
        let grain = u32::from_le_bytes(pbuf[0x0C..0x10].try_into().expect("fixed slice"));
        CONTEXTS.insert(
            handle,
            std::sync::Arc::new(ContextPace::new(frequency, grain)),
        );
    }
    if ctx.mem.write(out_context, &handle.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAudioOut2ContextDestroy(context)`: drop the context's pacing state and
/// any ports still bound to it (KytyPS5 clears the context's ports on destroy).
fn hle_context_destroy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    CONTEXTS.remove(&handle);
    PORTS.retain(|_, port| port.context != handle);
    OK
}

/// `sceAudioOut2ContextPush(context, blocking)`: submit every port of the
/// context whose PCM buffer is set (converted to interleaved-stereo f32 for the
/// host mixer), then pace the caller to one hardware grain. FMOD's PS5 output
/// path uses Push as its submission clock and never calls Advance; without
/// pacing the feeder outruns playback and starves the game. Submission is
/// fail-safe: an unknown context, a context with no ports, or a port with no
/// buffer just paces (or returns) with no audio, exactly as before.
fn hle_context_push(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    submit_context_ports(ctx.mem, handle);
    if let Some(pace) = CONTEXTS.get(&handle).map(|p| std::sync::Arc::clone(&p)) {
        pace.pace();
    }
    OK
}

/// `sceAudioOut2ContextAdvance(context)`: advancing renders one grain of audio
/// on hardware; pace to the same wall-clock cadence.
fn hle_context_advance(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    if let Some(pace) = CONTEXTS.get(&handle).map(|p| std::sync::Arc::clone(&p)) {
        pace.pace();
    }
    OK
}

/// `sceAudioOut2PortSetAttributes(port, attributes, num)`: capture the port's
/// PCM buffer pointer from the PCM attribute (KytyPS5 `ATTRIBUTE_ID_PCM` → an
/// `AudioOut2Pcm { const void* data }`), so `ContextPush` can play it.
/// Volume/routing attributes stay inert (SharpEmu's answer too). Always returns
/// OK so the mixer thread keeps running. Hardened per SharpEmu `e13cb28`: the
/// attribute count is bounded, every offset is checked for overflow, and each
/// 24-byte record and the 8-byte PCM pointer are read through bounds-checked
/// `GuestMemory` — a malformed array can neither overrun nor wedge the loop, it
/// just leaves the PCM pointer unchanged.
fn hle_port_set_attributes(ctx: &HleContext, args: &[u64]) -> u64 {
    let port = args.first().copied().unwrap_or(0);
    let attributes = args.get(1).copied().unwrap_or(0);
    let num = args.get(2).copied().unwrap_or(0).min(MAX_ATTRIBUTES);
    if port == 0 || attributes == 0 || num == 0 {
        return OK;
    }
    for i in 0..num {
        let Some(base) = i
            .checked_mul(ATTRIBUTE_STRIDE)
            .and_then(|off| attributes.checked_add(off))
        else {
            break;
        };
        let mut attr = [0u8; ATTRIBUTE_STRIDE as usize];
        if !ctx.mem.read(base, &mut attr) {
            break;
        }
        let attribute_id = u32::from_le_bytes(attr[0x00..0x04].try_into().expect("fixed slice"));
        let value = u64::from_le_bytes(attr[0x08..0x10].try_into().expect("fixed slice"));
        let value_size = u64::from_le_bytes(attr[0x10..0x18].try_into().expect("fixed slice"));
        if attribute_id == ATTRIBUTE_ID_PCM && value != 0 && value_size >= 8 {
            // value -> AudioOut2Pcm { const void* data }
            let mut data = [0u8; 8];
            if ctx.mem.read(value, &mut data) {
                if let Some(mut port_state) = PORTS.get_mut(&port) {
                    port_state.pcm_ptr = u64::from_le_bytes(data);
                }
            }
        }
    }
    OK
}

/// `sceAudioOut2ContextGetQueueLevel(context, outLevel)`: the push/advance
/// paths pace synchronously, so the queue is always drained — report level 0
/// (SharpEmu's answer). A null out-pointer is tolerated, matching SharpEmu.
fn hle_context_get_queue_level(ctx: &HleContext, args: &[u64]) -> u64 {
    let level = args.get(1).copied().unwrap_or(0);
    if level != 0 && !ctx.mem.write(level, &0u64.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// Per-context playback cadence, ported from SharpEmu's `ContextState`: one
/// grain is `grain_samples / frequency` seconds of wall-clock time, and each
/// paced call blocks until the previous grain's time has fully elapsed.
struct ContextPace {
    grain: std::time::Duration,
    /// Normalized frame count per grain (from the context param), reused as the
    /// per-push frame count when submitting a port's PCM (KytyPS5 `samples_num`).
    grain_samples: u32,
    next: std::sync::Mutex<std::time::Instant>,
}

impl ContextPace {
    fn new(frequency: u32, grain_samples: u32) -> Self {
        let frequency = u64::from(if frequency == 0 { 48000 } else { frequency });
        let grain_samples = if grain_samples == 0 {
            256
        } else {
            grain_samples
        };
        Self {
            grain: std::time::Duration::from_secs_f64(
                u64::from(grain_samples) as f64 / frequency as f64,
            ),
            grain_samples,
            next: std::sync::Mutex::new(std::time::Instant::now()),
        }
    }

    /// SharpEmu `PaceAdvance`: the first call after a quiet period starts a
    /// fresh grain without sleeping; a call inside an open grain sleeps out
    /// the remainder, so a mixer loop runs at the playback rate, not host speed.
    fn pace(&self) {
        let wait = {
            let mut next = self.next.lock().expect("pace mutex");
            let now = std::time::Instant::now();
            if *next <= now {
                *next = now + self.grain;
                return;
            }
            let wait = *next - now;
            *next += self.grain;
            wait
        };
        std::thread::sleep(wait);
    }
}

/// Per-context pacing state, keyed by the opaque context handle. Mirrors
/// SharpEmu's module-level `Contexts` dictionary.
static CONTEXTS: std::sync::LazyLock<dashmap::DashMap<u64, std::sync::Arc<ContextPace>>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// A port's PCM playback state, populated at `PortCreate` (format + owning
/// context) and `PortSetAttributes` (the guest PCM buffer pointer). Keyed by the
/// port handle. Mirrors KytyPS5's `AudioOut2PortStateEntry`.
#[derive(Clone, Copy)]
struct PortAudio {
    /// Owning context handle — the one `ContextPush` matches against.
    context: u64,
    /// Source channels per frame (1 mono, 2 stereo, 8 for 7.1, …).
    channels: u32,
    /// `true` for float32 samples, `false` for signed 16-bit.
    is_float: bool,
    /// Port sample rate (Hz), passed to the host mixer for resampling.
    sample_rate: u32,
    /// Frames submitted per push (the owning context's grain / `samples_num`).
    grain: u32,
    /// Guest PCM buffer pointer set via `PortSetAttributes` (0 = none yet).
    pcm_ptr: u64,
}

/// Per-port PCM state, keyed by port handle. Mirrors KytyPS5's
/// `g_audioout2_ports`.
static PORTS: std::sync::LazyLock<dashmap::DashMap<u64, PortAudio>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// Decode an `AudioOut2PortParam.data_format` into `(channels, is_float)`,
/// following KytyPS5 `audioout2_data_format_*`: channels are bits `[8,16)`
/// (0 → stereo, clamped to 16); the low 7 bits are the data type (0 = float32,
/// 1 = signed 16-bit).
fn decode_data_format(data_format: u32) -> (u32, bool) {
    let channels = {
        let c = (data_format >> 8) & 0xff;
        if c == 0 { 2 } else { c.min(MAX_CHANNELS) }
    };
    let is_float = (data_format & 0x7f) == 0;
    (channels, is_float)
}

/// Read a port's PCM format from its `AudioOut2PortParam` at `param`
/// (`+0x04 data_format`, `+0x08 sampling_freq`; KytyPS5 layout). A null or
/// unreadable pointer, or an out-of-range rate, falls back to stereo float
/// 48 kHz — the format never faults playback, it only picks a sane default.
fn read_port_format(mem: &dyn GuestMemory, param: u64) -> (u32, bool, u32) {
    let mut buf = [0u8; 0x10];
    if param == 0 || !mem.read(param, &mut buf) {
        return (2, true, 48_000);
    }
    let data_format = u32::from_le_bytes(buf[0x04..0x08].try_into().expect("fixed slice"));
    let sampling_freq = u32::from_le_bytes(buf[0x08..0x0C].try_into().expect("fixed slice"));
    let (channels, is_float) = decode_data_format(data_format);
    let sample_rate = if (8_000..=192_000).contains(&sampling_freq) {
        sampling_freq
    } else {
        48_000
    };
    (channels, is_float, sample_rate)
}

/// Read a port's current PCM buffer from guest memory and convert it to
/// interleaved-stereo f32 for [`raeen_audio::output::submit`]. Returns `None`
/// (no submission) when the port has no buffer, the size is implausible, or the
/// guest read fails — the caller then just paces, exactly as before.
///
/// The byte length (`grain × channels × bytes_per_sample`) is bounded to
/// [`MAX_PCM_BYTES`] and computed with checked arithmetic, so a malformed grain,
/// channel count, or format can never overrun the read (hardening mirrored from
/// SharpEmu `e13cb28`). Per-submission volume is unity: the host mixer applies
/// the user's master volume downstream in `output::fill`, and the AudioOut2
/// per-port volume/routing attributes are inert, so scaling here would
/// double-attenuate.
fn build_port_submission(mem: &dyn GuestMemory, port: &PortAudio) -> Option<(u32, Vec<f32>)> {
    if port.pcm_ptr == 0 {
        return None;
    }
    let channels = port.channels.clamp(1, MAX_CHANNELS) as usize;
    let bytes_per_sample = if port.is_float { 4 } else { 2 };
    let frames = port.grain as usize;
    let byte_len = frames
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(bytes_per_sample))?;
    if byte_len == 0 || byte_len > MAX_PCM_BYTES {
        return None;
    }
    let mut buf = vec![0u8; byte_len];
    if !mem.read(port.pcm_ptr, &mut buf) {
        return None;
    }
    let stereo =
        raeen_audio::pcm::convert_to_stereo_f32(&buf, frames, channels, port.is_float, 1.0);
    if stereo.is_empty() {
        return None;
    }
    Some((port.sample_rate, stereo))
}

/// Submit every port belonging to `context` that has a PCM buffer set. Collects
/// the matching ports up front so no port lock is held across the guest read /
/// host submit (KytyPS5's `audioout2_queue_context_audio`, adapted to Raeen's
/// f32 mixer).
fn submit_context_ports(mem: &dyn GuestMemory, context: u64) {
    let ports: Vec<PortAudio> = PORTS
        .iter()
        .filter(|e| e.value().context == context && e.value().pcm_ptr != 0)
        .map(|e| *e.value())
        .collect();
    for port in ports {
        if let Some((rate, stereo)) = build_port_submission(mem, &port) {
            raeen_audio::output::submit(rate, &stereo);
        }
    }
}

/// Resolve the owning context handle for a `PortCreate` call from its arguments.
///
/// The real PS5 ABI is `sceAudioOut2PortCreate(context, param, outPort)` (the
/// context in `rdi`); the SharpEmu-derived convention Raeen's lifecycle was
/// ported from instead reads `rdi` as a port *type* and takes the context in
/// `rcx`. Rather than commit to one register layout (this file cannot measure
/// GTA V), pick whichever argument is a currently-live context handle — the
/// real `rdi` first, then the SharpEmu `rcx`. If neither is live the port is
/// stored unlinked (`context = 0`) and simply never plays, matching prior
/// silent behaviour.
fn resolve_port_context(args: &[u64]) -> u64 {
    let candidates = [args.first().copied(), args.get(3).copied()];
    for handle in candidates.into_iter().flatten() {
        if handle != 0 && CONTEXTS.contains_key(&handle) {
            return handle;
        }
    }
    0
}

/// `sceAudioOut2PortCreate(type, param, outPort, context)`: returns a fresh
/// port handle encoding the type + a rolling 8-bit port id.
fn hle_port_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let ty = args.first().copied().unwrap_or(0) as i32;
    let param = args.get(1).copied().unwrap_or(0);
    let out_port = args.get(2).copied().unwrap_or(0);
    let context = args.get(3).copied().unwrap_or(0);
    if !(0..=255).contains(&ty) || param == 0 || out_port == 0 || context == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let port_id = (NEXT_PORT_ID.fetch_add(1, Ordering::Relaxed) as u32).wrapping_add(1) & 0xFF;
    let handle = 0x2000_0000u64 | ((ty as u32 as u64) << 16) | port_id as u64;

    // Record the port's PCM format (from its AudioOut2PortParam) and owning
    // context so `ContextPush` can interpret and submit its buffer. Additive:
    // it does not affect the handle or the existing validation/return path.
    let (channels, is_float, sample_rate) = read_port_format(ctx.mem, param);
    let context_handle = resolve_port_context(args);
    let grain = CONTEXTS
        .get(&context_handle)
        .map(|c| c.grain_samples)
        .unwrap_or(256);
    PORTS.insert(
        handle,
        PortAudio {
            context: context_handle,
            channels,
            is_float,
            sample_rate,
            grain,
            pcm_ptr: 0,
        },
    );

    if ctx.mem.write(out_port, &handle.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAudioOut2PortDestroy(port)`: drop the port's PCM state.
fn hle_port_destroy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    PORTS.remove(&handle);
    OK
}

/// `sceAudioOut2PortGetState(handle, state)`: fills a 0x20-byte state struct
/// derived from the port type encoded in `handle` (type 2 = a
/// personal/controller port: 1 channel; otherwise a main port: 2 channels).
fn hle_port_get_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let state = args.get(1).copied().unwrap_or(0);
    if handle == 0 || state == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let ty = ((handle >> 16) & 0xFF) as i32;
    let mut buf = [0u8; 0x20];
    let output: u16 = if ty == 2 { 0x40 } else { 0x01 };
    let channels: u8 = if ty == 2 { 1 } else { 2 };
    buf[0x00..0x02].copy_from_slice(&output.to_le_bytes());
    buf[0x02] = channels;
    buf[0x04..0x06].copy_from_slice(&(-1i16).to_le_bytes());
    if ctx.mem.write(state, &buf) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAudioOut2GetSpeakerInfo(info)`: fills a 0x40-byte speaker-info struct
/// (1 device, 2 channels, 48000 Hz).
fn hle_get_speaker_info(ctx: &HleContext, args: &[u64]) -> u64 {
    let info = args.first().copied().unwrap_or(0);
    if info == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 0x40];
    buf[0x00..0x04].copy_from_slice(&1u32.to_le_bytes());
    buf[0x04..0x08].copy_from_slice(&2u32.to_le_bytes());
    buf[0x08..0x0C].copy_from_slice(&48000u32.to_le_bytes());
    if ctx.mem.write(info, &buf) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAudioOut2UserCreate(userId, outUser)`: returns a fresh user handle for a
/// recognized user id (0, 1, 255, or 1000).
fn hle_user_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let user_id = args.first().copied().unwrap_or(0) as i32;
    let out_user = args.get(1).copied().unwrap_or(0);
    if !matches!(user_id, 0 | 1 | 255 | 1000) || out_user == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let handle = (NEXT_USER_HANDLE.fetch_add(1, Ordering::Relaxed) + 1) as u64;
    if ctx.mem.write(out_user, &handle.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x200),
            crate::TestAllocator::new(0),
        )
    }

    fn read_u32(mem: &crate::TestMemory, addr: u64) -> u32 {
        let mut b = [0u8; 4];
        assert!(mem.read(addr, &mut b));
        u32::from_le_bytes(b)
    }

    fn read_u64(mem: &crate::TestMemory, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(mem.read(addr, &mut b));
        u64::from_le_bytes(b)
    }

    #[test]
    fn reset_param_and_query_memory_and_speaker_info() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_context_reset_param(&ctx, &[0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_context_reset_param(&ctx, &[0x10]), OK);
        assert_eq!(read_u32(&mem, 0x10), CONTEXT_PARAM_SIZE as u32);
        assert_eq!(read_u32(&mem, 0x14), 2);
        assert_eq!(read_u32(&mem, 0x18), 48000);
        assert_eq!(read_u32(&mem, 0x1C), 0x400);

        assert_eq!(hle_context_query_memory(&ctx, &[0x10, 0x60]), OK);
        assert_eq!(read_u64(&mem, 0x60), CONTEXT_MEMORY_SIZE);
        assert_eq!(
            hle_context_query_memory(&ctx, &[0, 0x60]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        assert_eq!(hle_get_speaker_info(&ctx, &[0x80]), OK);
        assert_eq!(read_u32(&mem, 0x80), 1);
        assert_eq!(read_u32(&mem, 0x84), 2);
        assert_eq!(read_u32(&mem, 0x88), 48000);
    }

    #[test]
    fn context_create_returns_handle_and_validates() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Any zero argument is rejected.
        assert_eq!(
            hle_context_create(&ctx, &[0x10, 0x20, 0x1000, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // First handle is 2 (counter starts at 1, incremented before use).
        assert_eq!(hle_context_create(&ctx, &[0x10, 0x20, 0x1000, 0xA0]), OK);
        assert_eq!(read_u64(&mem, 0xA0), 2);
    }

    #[test]
    fn port_create_encodes_type_and_get_state() {
        NEXT_PORT_ID.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // type 0 (main), param/out/context non-null.
        assert_eq!(hle_port_create(&ctx, &[0, 0x20, 0xB0, 0x30]), OK);
        let handle = read_u64(&mem, 0xB0);
        assert_eq!(handle, 0x2000_0000 | 1); // type 0, port id 1
        // Out-of-range type rejected.
        assert_eq!(
            hle_port_create(&ctx, &[256, 0x20, 0xB0, 0x30]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        // A type-2 handle reports 1 channel; a type-0 handle reports 2.
        let handle_t2 = 0x2000_0000u64 | (2u64 << 16) | 5;
        assert_eq!(hle_port_get_state(&ctx, &[handle_t2, 0xC0]), OK);
        let mut buf = [0u8; 0x20];
        assert!(mem.read(0xC0, &mut buf));
        assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), 0x40);
        assert_eq!(buf[0x02], 1);
        assert_eq!(i16::from_le_bytes([buf[0x04], buf[0x05]]), -1);

        assert_eq!(hle_port_get_state(&ctx, &[handle, 0xD0]), OK);
        assert!(mem.read(0xD0, &mut buf));
        assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), 0x01);
        assert_eq!(buf[0x02], 2);
    }

    #[test]
    fn streaming_trio_is_registered_and_paces_to_the_grain() {
        let registry = HleRegistry::new();
        for name in [
            "sceAudioOut2ContextPush",
            "sceAudioOut2ContextAdvance",
            "sceAudioOut2PortSetAttributes",
        ] {
            assert!(
                registry.is_implemented("libSceAudioOut2", name),
                "libSceAudioOut2::{name} must be registered"
            );
        }

        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Attributes are accepted unconditionally (inert without a backend).
        assert_eq!(hle_port_set_attributes(&ctx, &[0x2000_0001, 0x40]), OK);
        // Unknown context handles are tolerated without sleeping.
        let t = std::time::Instant::now();
        assert_eq!(hle_context_push(&ctx, &[0xdead, 0x40, 0x100, 0]), OK);
        assert_eq!(hle_context_advance(&ctx, &[0xdead]), OK);
        assert!(t.elapsed() < std::time::Duration::from_millis(50));

        // A real context paces: grain=4800 samples @ 48000 Hz = 100 ms. The
        // first push starts the grain; the second must wait it out.
        let mut param = [0u8; CONTEXT_PARAM_SIZE];
        param[0x08..0x0C].copy_from_slice(&48000u32.to_le_bytes());
        param[0x0C..0x10].copy_from_slice(&4800u32.to_le_bytes());
        assert!(mem.write(0x80, &param));
        assert_eq!(hle_context_create(&ctx, &[0x80, 0x100, 0x1000, 0x180]), OK);
        let mut hbuf = [0u8; 8];
        assert!(mem.read(0x180, &mut hbuf));
        let handle = u64::from_le_bytes(hbuf);
        assert_eq!(hle_context_push(&ctx, &[handle, 0x40, 0x100, 0]), OK);
        let t = std::time::Instant::now();
        assert_eq!(hle_context_push(&ctx, &[handle, 0x40, 0x100, 0]), OK);
        assert!(
            t.elapsed() >= std::time::Duration::from_millis(90),
            "second push inside the same grain must wait it out ({:?})",
            t.elapsed()
        );
        // Destroy drops the pacing state; a later push is unpaced again.
        assert_eq!(hle_context_destroy(&ctx, &[handle]), OK);
        assert!(CONTEXTS.get(&handle).is_none());

        // Queue level: always drained (synchronous pacing), null out tolerated.
        assert!(mem.write(0x190, &u64::MAX.to_le_bytes()));
        assert_eq!(hle_context_get_queue_level(&ctx, &[handle, 0x190]), OK);
        let mut lvl = [0u8; 8];
        assert!(mem.read(0x190, &mut lvl));
        assert_eq!(u64::from_le_bytes(lvl), 0);
        assert_eq!(hle_context_get_queue_level(&ctx, &[handle, 0]), OK);
    }

    #[test]
    fn user_create_validates_user_id() {
        NEXT_USER_HANDLE.store(1, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_user_create(&ctx, &[7, 0xE0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_user_create(&ctx, &[1000, 0xE0]), OK);
        assert_eq!(read_u64(&mem, 0xE0), 2);
    }

    fn big_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x400),
            crate::TestAllocator::new(0),
        )
    }

    fn write_u32(mem: &crate::TestMemory, addr: u64, v: u32) {
        assert!(mem.write(addr, &v.to_le_bytes()));
    }
    fn write_u64(mem: &crate::TestMemory, addr: u64, v: u64) {
        assert!(mem.write(addr, &v.to_le_bytes()));
    }
    fn write_f32(mem: &crate::TestMemory, addr: u64, v: f32) {
        assert!(mem.write(addr, &v.to_le_bytes()));
    }

    /// Build a context (grain=2 @ 48 kHz) and a float-stereo port linked to it,
    /// register a PCM buffer via SetAttributes, and return `(port_handle,
    /// ctx_handle, pcm_buffer_addr)`. Uses the SharpEmu register convention
    /// (`type` in arg0, context in arg3) so the port passes the lifecycle's
    /// existing `0..=255` type validation regardless of the context handle
    /// value — [`resolve_port_context`] then links via arg3.
    fn setup_playing_port(ctx: &HleContext, mem: &crate::TestMemory) -> (u64, u64, u64) {
        // Context param: frequency @ +0x08 = 48000, grain @ +0x0C = 2.
        write_u32(mem, 0x28, 48_000);
        write_u32(mem, 0x2C, 2);
        assert_eq!(hle_context_create(ctx, &[0x20, 0x300, 0x100, 0x60]), OK);
        let ctx_handle = read_u64(mem, 0x60);

        // AudioOut2PortParam @ 0x70: data_format @ +0x04 = 0x200 (float, 2ch),
        // sampling_freq @ +0x08 = 48000.
        write_u32(mem, 0x74, 0x200);
        write_u32(mem, 0x78, 48_000);
        // type=0 (arg0), param=0x70, outPort=0x90, context=ctx_handle (arg3).
        assert_eq!(hle_port_create(ctx, &[0, 0x70, 0x90, ctx_handle]), OK);
        let port_handle = read_u64(mem, 0x90);

        // PCM buffer @ 0xD0: two float-stereo frames.
        let pcm = 0xD0u64;
        write_f32(mem, pcm, 0.5);
        write_f32(mem, pcm + 4, -0.5);
        write_f32(mem, pcm + 8, 0.25);
        write_f32(mem, pcm + 12, -0.25);

        // AudioOut2Pcm { const void* data } @ 0xC0.
        write_u64(mem, 0xC0, pcm);
        // AudioOut2Attribute[0] @ 0xA0: { id=0 (PCM), reserved, value=0xC0,
        // value_size=8 }.
        write_u32(mem, 0xA0, ATTRIBUTE_ID_PCM);
        write_u32(mem, 0xA4, 0);
        write_u64(mem, 0xA8, 0xC0);
        write_u64(mem, 0xB0, 8);
        assert_eq!(hle_port_set_attributes(ctx, &[port_handle, 0xA0, 1]), OK);

        (port_handle, ctx_handle, pcm)
    }

    #[test]
    fn port_create_records_float_stereo_format_and_context_link() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        NEXT_PORT_ID.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = big_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let (port_handle, ctx_handle, _pcm) = setup_playing_port(&ctx, &mem);

        let port = *PORTS.get(&port_handle).expect("port audio state stored");
        assert_eq!(port.context, ctx_handle, "port linked to its context");
        assert_eq!(port.channels, 2);
        assert!(port.is_float);
        assert_eq!(port.sample_rate, 48_000);
        assert_eq!(port.grain, 2, "grain taken from the owning context");
    }

    #[test]
    fn set_attributes_captures_pcm_pointer() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        NEXT_PORT_ID.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = big_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let (port_handle, _ctx_handle, pcm) = setup_playing_port(&ctx, &mem);

        let port = *PORTS.get(&port_handle).expect("port audio state stored");
        assert_eq!(
            port.pcm_ptr, pcm,
            "PCM attribute captured the buffer pointer"
        );
    }

    #[test]
    fn push_converts_port_pcm_to_stereo_f32_and_submits() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        NEXT_PORT_ID.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = big_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let (port_handle, ctx_handle, _pcm) = setup_playing_port(&ctx, &mem);

        // The exact stereo-f32 that Push hands to the host mixer.
        let port = *PORTS.get(&port_handle).expect("port audio state stored");
        let (rate, stereo) =
            build_port_submission(&mem, &port).expect("a port with PCM produces a submission");
        assert_eq!(rate, 48_000);
        assert_eq!(stereo, vec![0.5, -0.5, 0.25, -0.25]);

        // Push drives that same path into raeen_audio::output::submit.
        let before = raeen_audio::output::submit_call_count();
        assert_eq!(hle_context_push(&ctx, &[ctx_handle, 0]), OK);
        let after = raeen_audio::output::submit_call_count();
        assert!(
            after >= before + 1,
            "push must invoke the stereo-f32 submit path ({before} -> {after})"
        );
    }

    #[test]
    fn push_without_a_pcm_buffer_does_not_submit() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        NEXT_PORT_ID.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = big_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Context + port, but no SetAttributes: pcm_ptr stays 0.
        write_u32(&mem, 0x28, 48_000);
        write_u32(&mem, 0x2C, 2);
        assert_eq!(hle_context_create(&ctx, &[0x20, 0x300, 0x100, 0x60]), OK);
        let ctx_handle = read_u64(&mem, 0x60);
        write_u32(&mem, 0x74, 0x200);
        write_u32(&mem, 0x78, 48_000);
        assert_eq!(hle_port_create(&ctx, &[0, 0x70, 0x90, ctx_handle]), OK);
        let port_handle = read_u64(&mem, 0x90);

        let port = *PORTS.get(&port_handle).expect("port audio state stored");
        assert_eq!(port.pcm_ptr, 0);
        assert!(
            build_port_submission(&mem, &port).is_none(),
            "a port with no PCM buffer yields no submission"
        );
        // Push still succeeds and paces without playing.
        assert_eq!(hle_context_push(&ctx, &[ctx_handle, 0]), OK);
    }

    #[test]
    fn build_submission_bounds_an_absurd_grain() {
        let (kernel, mem, alloc) = big_env();
        let _ctx = test_ctx(&kernel, &mem, &alloc);
        // A wild grain/format must not overrun: grain * 16ch * 4B dwarfs
        // MAX_PCM_BYTES, so no read is attempted.
        let port = PortAudio {
            context: 0,
            channels: MAX_CHANNELS,
            is_float: true,
            sample_rate: 48_000,
            grain: u32::MAX,
            pcm_ptr: 0xD0,
        };
        assert!(build_port_submission(&mem, &port).is_none());
    }

    #[test]
    fn set_attributes_tolerates_a_wild_attribute_pointer() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        NEXT_PORT_ID.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = big_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        write_u32(&mem, 0x28, 48_000);
        write_u32(&mem, 0x2C, 2);
        assert_eq!(hle_context_create(&ctx, &[0x20, 0x300, 0x100, 0x60]), OK);
        let ctx_handle = read_u64(&mem, 0x60);
        write_u32(&mem, 0x74, 0x200);
        write_u32(&mem, 0x78, 48_000);
        assert_eq!(hle_port_create(&ctx, &[0, 0x70, 0x90, ctx_handle]), OK);
        let port_handle = read_u64(&mem, 0x90);

        // Attribute array pointer is out of range: the read fails, the loop
        // breaks, the call still returns OK and leaves the PCM pointer at 0.
        assert_eq!(
            hle_port_set_attributes(&ctx, &[port_handle, 0xFFFF_0000, 4]),
            OK
        );
        let port = *PORTS.get(&port_handle).expect("port audio state stored");
        assert_eq!(port.pcm_ptr, 0);
    }

    #[test]
    fn resolve_port_context_links_via_either_abi() {
        NEXT_CONTEXT_HANDLE.store(1, Ordering::Relaxed);
        let (kernel, mem, alloc) = big_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        write_u32(&mem, 0x28, 48_000);
        write_u32(&mem, 0x2C, 2);
        assert_eq!(hle_context_create(&ctx, &[0x20, 0x300, 0x100, 0x60]), OK);
        let ctx_handle = read_u64(&mem, 0x60);

        // Real ABI: context in arg0.
        assert_eq!(
            resolve_port_context(&[ctx_handle, 0x70, 0x90, 0]),
            ctx_handle
        );
        // SharpEmu ABI: context in arg3.
        assert_eq!(
            resolve_port_context(&[0, 0x70, 0x90, ctx_handle]),
            ctx_handle
        );
        // Neither argument is a live context: unlinked.
        assert_eq!(resolve_port_context(&[0xDEAD, 0x70, 0x90, 0xBEEF]), 0);
    }
}
