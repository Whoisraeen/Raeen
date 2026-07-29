//! HLE libSceSystemService — system state / events / settings.
//!
//! A running title polls `sceSystemServiceGetStatus` every frame to learn
//! about system events (a return to the home menu, an overlay, a controller
//! change) and reads system settings via `sceSystemServiceParamGetInt`.
//! Raeen reports a quiet, steady state (no pending events, full display safe
//! area) so a title's main loop runs undisturbed. Struct sizes, the safe-area
//! `1.0` ratio, and the `ParamGetInt` value mapping are cross-checked against
//! SharpEmu's `SystemServiceExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_SYSTEM_SERVICE_ERROR_PARAMETER`.
const ERROR_PARAMETER: u64 = 0x80A1_0003;
/// `SCE_SYSTEM_SERVICE_ERROR_NO_EVENT` — the event queue is empty. shadPS4
/// `systemservice_error.h`.
const ERROR_NO_EVENT: u64 = 0x80A1_0004;
/// `SceSystemServiceStatus` is a 12-byte struct; the first `int32` is
/// `eventNum` (pending system events).
const STATUS_SIZE: usize = 0x0C;
/// `SceSystemServiceDisplaySafeAreaInfo` = `float ratio` + 128 reserved
/// bytes.
const SAFE_AREA_SIZE: usize = 4 + 128;

/// Register libSceSystemService HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceSystemService",
        "sceSystemServiceGetStatus",
        hle_get_status,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceParamGetInt",
        hle_param_get_int,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceGetDisplaySafeAreaInfo",
        hle_get_safe_area,
    );
    // `sceSystemServiceGetHdrToneMapLuminance()` takes no arguments and simply
    // reports success: Raeen presents an SDR display, so there is no HDR
    // tone-map curve to report. shadPS4 stubs it identically
    // (`systemservice.cpp:1747`). Measured: Until Dawn stops its boot here.
    registry.register(
        "libSceSystemService",
        "sceSystemServiceGetHdrToneMapLuminance",
        hle_ok,
    );
    // Notice-screen skip flag. Reporting 0 (do not skip) is the neutral answer
    // and matches shadPS4, which stubs all three of these
    // (`scripts/aerolib.inl`: 3RQ5aQfnstU / Q3utJvma4Mo / 8Lo6Zv94aho). The
    // setters are registered alongside the getter because a title that queries
    // the flag during boot generally sets it a moment later.
    // Measured: ASTRO.BOT stops its boot on the getter.
    for f in [
        "sceSystemServiceGetNoticeScreenSkipFlag",
        "sceSystemServiceSetNoticeScreenSkipFlag",
        "sceSystemServiceDisableNoticeScreenSkipFlagAutoSet",
    ] {
        registry.register("libSceSystemService", f, hle_ok);
    }
    registry.register(
        "libSceSystemService",
        "sceSystemServiceHideSplashScreen",
        hle_hide_splash_screen,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceReportAbnormalTermination",
        hle_ok,
    );
    // The per-frame system-event pump. Minecraft's main loop calls this every
    // tick (measured: it appears in the steady-state HLE poll set); when it was
    // UNREGISTERED the call hit an unresolved-import error rather than the
    // defined "nothing pending" answer. shadPS4 (systemservice.cpp:1984)
    // returns ERROR_NO_EVENT when its event queue is empty and pops a real
    // event otherwise. With no event source wired yet, the honest answer is
    // "queue empty" — a valid, non-error state the caller handles by moving on,
    // NOT a failure it retries. This is the correct baseline; a real event
    // source (e.g. a game-intent launch event) is a later, separate step.
    registry.register(
        "libSceSystemService",
        "sceSystemServiceReceiveEvent",
        hle_receive_event,
    );
    // Player (profile) dialog overlay (measured GTA V imports). The param
    // initializer only touches caller memory whose layout is undocumented, so
    // it is acknowledged without writing; Launch has no host overlay to show,
    // so it is refused with the documented PARAMETER error — the title treats
    // the profile popup as unavailable and continues, rather than waiting on
    // an overlay that can never close.
    registry.register_incomplete(
        "libSceSystemService",
        "sceSystemServiceInitializePlayerDialogParam",
        hle_ok,
        "reports success without initializing the caller's player-dialog param out-struct",
    );
    registry.register_incomplete(
        "libSceSystemService",
        "sceSystemServiceLaunchPlayerDialog",
        hle_launch_player_dialog,
        "no host overlay: the player dialog refuses to launch",
    );
}

/// `sceSystemServiceLaunchPlayerDialog(param)`: no overlay exists to display.
fn hle_launch_player_dialog(_ctx: &HleContext, args: &[u64]) -> u64 {
    tracing::debug!(
        "sceSystemServiceLaunchPlayerDialog(param={:#x}) -> ERROR_PARAMETER (no host overlay)",
        args.first().copied().unwrap_or(0)
    );
    ERROR_PARAMETER
}

/// `sceSystemServiceHideSplashScreen()`: the title declares its own rendering
/// ready, so the system boot splash (the package's `sce_sys/pic0.png`, shown
/// since launch) comes down. This is a real presentation transition, not a
/// status stub — on hardware the splash persists until exactly this call.
fn hle_hide_splash_screen(ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceSystemServiceHideSplashScreen()");
    ctx.gpu.hide_splash();
    SCE_OK
}

/// `sceSystemServiceReceiveEvent(SceSystemServiceEvent *event)`: report an
/// empty queue. shadPS4 returns `ERROR_NO_EVENT` (0x80A10004) with the event
/// untouched when nothing is pending; the caller treats it as "no work this
/// frame", not an error to retry on.
fn hle_receive_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let event_ptr = args.first().copied().unwrap_or(0);
    debug!("sceSystemServiceReceiveEvent(event={event_ptr:#x})");
    // RE probe (RAEEN_TRACE_MAINLOOP): ReceiveEvent is polled from the main
    // loop's per-frame tick, so its caller return-addr points INTO that tick —
    // the state machine that decides whether to create the Gameface view. Log
    // it once so `raeen --disas` can be aimed at the exact function.
    if std::env::var_os("RAEEN_TRACE_MAINLOOP").is_some() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if !LOGGED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                caller = format_args!("{:#x}", ctx.caller_return_addr),
                "TRACE_MAINLOOP: sceSystemServiceReceiveEvent called from (main-loop tick)"
            );
        }
    }
    if event_ptr == 0 {
        return ERROR_PARAMETER;
    }
    ERROR_NO_EVENT
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceSystemServiceGetStatus(SceSystemServiceStatus *status)`: reports a
/// quiet state — `eventNum = 0`, no overlay, not backgrounded (all-zero
/// 12-byte struct) — so a title's per-frame poll sees nothing to handle.
fn hle_get_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status_ptr = args.first().copied().unwrap_or(0);
    debug!("sceSystemServiceGetStatus(status={status_ptr:#x})");
    // RE confirmation probe (RAEEN_TRACE_UI): GetStatus is polled every frame,
    // so it is a stand-in for the per-frame HBUI/router tick. Read the gate a
    // navigate-gate RE pinned: P = *[0xE15B830] (app shared-state singleton),
    // flag = byte[P+0x248]. The router tick at eboot 0x1112794 returns BEFORE
    // any screen registration/navigation while this byte is non-zero. Log it
    // on change + first few reads to see whether it is stuck non-zero (= the
    // menu never navigates) or clears (gate is downstream).
    if std::env::var_os("RAEEN_TRACE_UI").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        const BASE: u64 = 0x0000_1000_0000_0000;
        static LAST: AtomicU32 = AtomicU32::new(0xFFFF_FFFF);
        static SEEN: AtomicU32 = AtomicU32::new(0);
        let rd = |a: u64| -> Option<u64> {
            let mut w = [0u8; 8];
            ctx.mem.read(a, &mut w).then(|| u64::from_le_bytes(w))
        };
        let p = rd(BASE + 0xE15_B830).unwrap_or(0);
        let flag = if p != 0 {
            let mut b = [0u8; 1];
            ctx.mem.read(p + 0x248, &mut b).then_some(b[0])
        } else {
            None
        };
        let cur = flag.map_or(0xFFFF_FFFEu32, u32::from);
        let n = SEEN.fetch_add(1, Ordering::Relaxed);
        if LAST.swap(cur, Ordering::Relaxed) != cur || n < 4 {
            warn!(
                singleton = format_args!("{:#x}", p.wrapping_sub(BASE)),
                gate_byte_0x248 = ?flag,
                frame = n,
                "TRACE_UI: navigate-gate G1 flag (byte[P+0x248]) — non-zero = menu blocked"
            );
        }
    }
    if status_ptr == 0 {
        return ERROR_PARAMETER;
    }
    if !ctx.mem.write(status_ptr, &[0u8; STATUS_SIZE]) {
        warn!("sceSystemServiceGetStatus: status out-ptr {status_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// `sceSystemServiceParamGetInt(int paramId, int *value)`: writes the system
/// setting for `paramId`. Values mirror SharpEmu's mapping (a stable set of
/// defaults): params 1/2/3/1000 → 1, param 4 → 180, everything else → 0.
fn hle_param_get_int(ctx: &HleContext, args: &[u64]) -> u64 {
    let param_id = args.first().copied().unwrap_or(0) as i32;
    let value_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceSystemServiceParamGetInt(paramId={param_id}, value={value_ptr:#x})");
    if value_ptr == 0 {
        return ERROR_PARAMETER;
    }
    let value: i32 = match param_id {
        1 | 2 | 3 | 1000 => 1,
        4 => 180,
        _ => 0,
    };
    if !ctx.mem.write(value_ptr, &value.to_le_bytes()) {
        warn!("sceSystemServiceParamGetInt: value out-ptr {value_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// `sceSystemServiceGetDisplaySafeAreaInfo(SceSystemServiceDisplaySafeAreaInfo
/// *info)`: reports a full safe area (`ratio = 1.0`) so a title renders to
/// the whole display.
fn hle_get_safe_area(ctx: &HleContext, args: &[u64]) -> u64 {
    let info_ptr = args.first().copied().unwrap_or(0);
    debug!("sceSystemServiceGetDisplaySafeAreaInfo(info={info_ptr:#x})");
    if info_ptr == 0 {
        return ERROR_PARAMETER;
    }
    let mut buf = [0u8; SAFE_AREA_SIZE];
    buf[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // ratio = 1.0 (full)
    if !ctx.mem.write(info_ptr, &buf) {
        warn!("sceSystemServiceGetDisplaySafeAreaInfo: info out-ptr {info_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn get_status_reports_no_events() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, &[0xEE; STATUS_SIZE]));
        assert_eq!(hle_get_status(&ctx, &[0x100]), SCE_OK);
        let mut s = [0u8; STATUS_SIZE];
        assert!(mem.read(0x100, &mut s));
        assert_eq!(
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
            0,
            "eventNum == 0"
        );
        assert_eq!(hle_get_status(&ctx, &[0]), ERROR_PARAMETER);
    }

    #[test]
    fn param_get_int_mirrors_the_default_mapping() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        for (pid, want) in [(1, 1), (2, 1), (3, 1), (1000, 1), (4, 180), (99, 0)] {
            assert_eq!(hle_param_get_int(&ctx, &[pid as u64, 0x200]), SCE_OK);
            let mut v = [0u8; 4];
            assert!(mem.read(0x200, &mut v));
            assert_eq!(i32::from_le_bytes(v), want, "paramId {pid}");
        }
        assert_eq!(hle_param_get_int(&ctx, &[1, 0]), ERROR_PARAMETER);
    }

    #[test]
    fn safe_area_reports_full_ratio() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_safe_area(&ctx, &[0x100]), SCE_OK);
        let mut r = [0u8; 4];
        assert!(mem.read(0x100, &mut r));
        assert_eq!(f32::from_le_bytes(r), 1.0, "full safe-area ratio");
    }

    #[test]
    fn receive_event_reports_empty_queue_not_error() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A null out-pointer is a parameter error (shadPS4 parity).
        assert_eq!(hle_receive_event(&ctx, &[0]), ERROR_PARAMETER);
        // A valid pointer with nothing queued reports NO_EVENT — the defined
        // "no work this frame" answer, NOT a generic failure the title retries.
        assert_eq!(hle_receive_event(&ctx, &[0x100]), ERROR_NO_EVENT);
    }

    #[test]
    fn receive_event_is_registered() {
        let registry = HleRegistry::new();
        register(&registry);
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // A resolved call returns Some(NO_EVENT) for a valid pointer; an
        // unregistered function would return None (unresolved import).
        assert_eq!(
            registry.call(
                &ctx,
                "libSceSystemService",
                "sceSystemServiceReceiveEvent",
                &[0x100],
            ),
            Some(ERROR_NO_EVENT),
            "the per-frame event pump must resolve, not hit an unresolved import"
        );
    }
}
