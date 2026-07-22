//! HLE libSceAudioOut2 — the PS5 (Gen5) AudioOut2 context/port lifecycle.
//!
//! A faithful Rust port of SharpEmu's `AudioOut2Exports` (GPL-2.0).
//! `libSceAudioOut2` is the Gen5 audio-output API (distinct from the older
//! `libSceAudioOut`, ported in [`crate::libsce_audio_out`]). Raeen has no audio
//! device backend, so nothing is ever played — but the context/port
//! **bookkeeping** is real: reset-param / query-memory / speaker-info fill the
//! guest structs the title reads, and create calls hand back opaque handles so
//! audio init completes instead of hanging.
//!
//! Struct sizes/offsets are ported verbatim from SharpEmu, whose
//! `AudioOut2ContextParamSize` (0x30) note records real guest evidence (Quake's
//! stack canary at param+0x60): only the populated prefix is written, well
//! below the canary. Handles/port ids are module-level monotonic counters
//! matching SharpEmu's statics. `OrbisGen2Result` codes map to the real Orbis
//! `EINVAL`/`EFAULT` (`0x8002_0016`/`0x8002_000E`) as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

const CONTEXT_PARAM_SIZE: usize = 0x30;
const CONTEXT_MEMORY_SIZE: u64 = 0x10000;

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
    registry.register("libSceAudioOut2", "sceAudioOut2PortDestroy", |_, _| OK);
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

/// `sceAudioOut2ContextDestroy(context)`: drop the context's pacing state.
fn hle_context_destroy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    CONTEXTS.remove(&handle);
    OK
}

/// `sceAudioOut2ContextPush(context, pcm, frames, format?)`: accept the frames
/// (discarded — no audio backend) and pace the caller to one hardware grain.
/// FMOD's PS5 output path uses Push as its submission clock and never calls
/// Advance; without pacing the feeder outruns playback and starves the game.
fn hle_context_push(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
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

/// `sceAudioOut2PortSetAttributes(port, attributes)`: volume/routing
/// attributes are inert without an audio backend — accept (SharpEmu's answer
/// too) so the mixer thread keeps running.
fn hle_port_set_attributes(_ctx: &HleContext, _args: &[u64]) -> u64 {
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
    next: std::sync::Mutex<std::time::Instant>,
}

impl ContextPace {
    fn new(frequency: u32, grain_samples: u32) -> Self {
        let frequency = u64::from(if frequency == 0 { 48000 } else { frequency });
        let grain_samples = u64::from(if grain_samples == 0 {
            256
        } else {
            grain_samples
        });
        Self {
            grain: std::time::Duration::from_secs_f64(grain_samples as f64 / frequency as f64),
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
    if ctx.mem.write(out_port, &handle.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
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
    use crate::{GuestMemory, test_ctx};

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
}
