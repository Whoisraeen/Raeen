//! Native Sony controller reader over raw HID (Windows).
//!
//! Clean-room port of SharpEmu's `WindowsDualSenseReader.cs` +
//! `WindowsHidNative.cs`, extended to the v0.0.3 SDL controller behavior from
//! PR #670 (GPL-2.0-or-later, © SharpEmu Emulator Project).
//! Zero external dependencies: the device is found by `SetupDi*` enumeration
//! filtered on Sony's VID/PID, opened with `CreateFileW`, and read with
//! `ReadFile`; the fixed input report is parsed at documented offsets. Handles
//! DualSense/Edge and both DualShock 4 revisions over USB and Bluetooth.
//!
//! Input is implemented for both pad generations. Rumble output remains
//! DualSense-only; DS4 output, advanced haptics / adaptive triggers, and
//! lightbar output remain follow-ups. Output reports go out on a dedicated
//! writer thread over a second handle to the same device path (SharpEmu's
//! design), so the blocking reader never serializes against a write. The
//! transport (USB vs Bluetooth) is detected from the first parsed DualSense
//! report; Bluetooth output frames carry the required CRC-32.
//!
//! The pure [`parse_report`] / [`build_output_report`] functions carry the
//! whole mapping and are unit-tested without a device; the Windows FFI in
//! `imp` only feeds them.

use crate::ControllerState;

/// Sony Corp. USB vendor id.
const SONY_VID: u16 = 0x054C;
/// DualSense (CFI-ZCT1) product id.
const DUALSENSE_PID: u16 = 0x0CE6;
/// DualSense Edge (CFI-ZCT1) product id.
const DUALSENSE_EDGE_PID: u16 = 0x0DF2;
/// DualShock 4 first revision (CUH-ZCT1) product id.
const DUALSHOCK4_V1_PID: u16 = 0x05C4;
/// DualShock 4 second revision (CUH-ZCT2) product id.
const DUALSHOCK4_V2_PID: u16 = 0x09CC;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SonyControllerKind {
    DualSense,
    DualShock4,
}

fn sony_controller_kind(product_id: u16) -> Option<SonyControllerKind> {
    match product_id {
        DUALSENSE_PID | DUALSENSE_EDGE_PID => Some(SonyControllerKind::DualSense),
        DUALSHOCK4_V1_PID | DUALSHOCK4_V2_PID => Some(SonyControllerKind::DualShock4),
        _ => None,
    }
}

/// Decode the 4-bit hat/D-pad value (`0`=N … `7`=NW, `8`=centered) into the
/// four discrete D-pad booleans, as `(up, down, left, right)`.
fn hat_to_dpad(hat: u8) -> (bool, bool, bool, bool) {
    match hat & 0x0F {
        0 => (true, false, false, false),  // N
        1 => (true, false, false, true),   // NE
        2 => (false, false, false, true),  // E
        3 => (false, true, false, true),   // SE
        4 => (false, true, false, false),  // S
        5 => (false, true, true, false),   // SW
        6 => (false, false, true, false),  // W
        7 => (true, false, true, false),   // NW
        _ => (false, false, false, false), // 8 (centered) / invalid
    }
}

/// Map an unsigned stick byte (`0`=min, `128`=center, `255`=max, Y growing
/// downward) to this crate's `-1.0..=1.0` encoding — no inversion, since a
/// DualSense byte of `0` and this crate's `-1.0` both mean "up/left".
fn byte_to_axis(b: u8) -> f32 {
    ((b as f32 - 128.0) / 127.0).clamp(-1.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn sony_controller_state(
    left_x: u8,
    left_y: u8,
    right_x: u8,
    right_y: u8,
    l2: u8,
    r2: u8,
    buttons0: u8,
    buttons1: u8,
    buttons2: u8,
) -> ControllerState {
    let (dpad_up, dpad_down, dpad_left, dpad_right) = hat_to_dpad(buttons0);

    ControllerState {
        square: buttons0 & 0x10 != 0,
        cross: buttons0 & 0x20 != 0,
        circle: buttons0 & 0x40 != 0,
        triangle: buttons0 & 0x80 != 0,
        l1: buttons1 & 0x01 != 0,
        r1: buttons1 & 0x02 != 0,
        // buttons1 bits 0x04 / 0x08 are the digital L2/R2; use the analog
        // axes so both supported pad generations share one trigger contract.
        create: buttons1 & 0x10 != 0,
        options: buttons1 & 0x20 != 0,
        l3: buttons1 & 0x40 != 0,
        r3: buttons1 & 0x80 != 0,
        ps_button: buttons2 & 0x01 != 0,
        touchpad_click: buttons2 & 0x02 != 0,
        dpad_up,
        dpad_down,
        dpad_left,
        dpad_right,
        left_stick_x: byte_to_axis(left_x),
        left_stick_y: byte_to_axis(left_y),
        right_stick_x: byte_to_axis(right_x),
        right_stick_y: byte_to_axis(right_y),
        l2_trigger: l2 as f32 / 255.0,
        r2_trigger: r2 as f32 / 255.0,
        ..Default::default()
    }
}

/// Parse a DualSense HID input report into a [`ControllerState`].
///
/// Accepts USB report id `0x01` (payload at byte 1) and Bluetooth extended
/// report id `0x31` (payload at byte 2); any other or too-short report yields
/// `None`. Button offsets follow the layout shared with Linux
/// `hid-playstation`: `buttons0` carries the face buttons + hat, `buttons1`
/// the shoulders / Create / Options / thumb-clicks, `buttons2` the PS button
/// and touchpad click. L2/R2 are taken from the analog trigger axes (the
/// pipeline derives the digital bit at `>0.5`), matching the other backends.
#[must_use]
pub fn parse_report(report: &[u8]) -> Option<ControllerState> {
    let offset = if report.len() >= 11 && report[0] == 0x01 {
        1usize
    } else if report.len() >= 12 && report[0] == 0x31 {
        2usize
    } else {
        return None;
    };
    // We read through offset + 9 (buttons2); guard the upper bound too.
    if report.len() < offset + 10 {
        return None;
    }

    let left_x = report[offset];
    let left_y = report[offset + 1];
    let right_x = report[offset + 2];
    let right_y = report[offset + 3];
    let l2 = report[offset + 4];
    let r2 = report[offset + 5];
    let buttons0 = report[offset + 7];
    let buttons1 = report[offset + 8];
    let buttons2 = report[offset + 9];

    Some(sony_controller_state(
        left_x, left_y, right_x, right_y, l2, r2, buttons0, buttons1, buttons2,
    ))
}

/// Parse a native DualShock 4 HID input report.
///
/// The DS4 uses report id `0x01` over USB and for its minimal Bluetooth report,
/// with the common payload beginning at byte 1. Its full Bluetooth report is id
/// `0x11`; two transport bytes precede the same payload, so it begins at byte 3.
/// The common bytes are four stick axes, three button bytes, then the two analog
/// triggers. DS4 Share maps to the guest's Create button and its touchpad click
/// remains distinct.
#[must_use]
pub fn parse_dualshock4_report(report: &[u8]) -> Option<ControllerState> {
    let offset = if report.len() >= 10 && report[0] == 0x01 {
        1usize
    } else if report.len() >= 12 && report[0] == 0x11 {
        3usize
    } else {
        return None;
    };
    if report.len() < offset + 9 {
        return None;
    }

    Some(sony_controller_state(
        report[offset],
        report[offset + 1],
        report[offset + 2],
        report[offset + 3],
        report[offset + 7],
        report[offset + 8],
        report[offset + 4],
        report[offset + 5],
        report[offset + 6],
    ))
}

/// One step of the standard reflected CRC-32 (poly `0xEDB88320`), ported from
/// SharpEmu `WindowsDualSenseReader.Crc32Update`.
fn crc32_update(mut crc: u32, value: u8) -> u32 {
    crc ^= value as u32;
    for _ in 0..8 {
        crc = (crc >> 1) ^ (0xEDB8_8320 & ((crc & 1).wrapping_neg()));
    }
    crc
}

/// CRC-32 over a seed byte followed by `data` — the DualSense Bluetooth
/// output-report checksum, which is seeded with `0xA2` (the HID "output
/// report" BT header byte that is not itself part of the report buffer).
/// Ported from SharpEmu `WindowsDualSenseReader.Crc32`.
#[must_use]
fn output_crc32(seed: u8, data: &[u8]) -> u32 {
    let mut crc = crc32_update(0xFFFF_FFFF, seed);
    for &value in data {
        crc = crc32_update(crc, value);
    }
    !crc
}

/// Build a DualSense rumble output report (SharpEmu
/// `BuildOutputReportLocked`, rumble-only subset; offsets are the layout
/// shared with Linux `hid-playstation`).
///
/// The 47-byte common payload sets `valid_flag0 = 0x03` (compatible-vibration
/// plus haptics-select — required for classic rumble on the haptics-native
/// DualSense), leaves `valid_flag1 = 0` so the lightbar / player LEDs are
/// untouched, and carries `motor_right` (small/weak) then `motor_left`
/// (large/strong). USB wraps it as report id `0x02` (48 bytes); Bluetooth as
/// report id `0x31` with a 4-bit rolling sequence tag, a `0x10` data tag, and
/// a trailing CRC-32 over `0xA2` plus the first 74 bytes (78 bytes total —
/// without the CRC the pad silently discards the frame).
#[must_use]
pub fn build_output_report(bluetooth: bool, bt_seq: u8, large: u8, small: u8) -> Vec<u8> {
    let mut common = [0u8; 47];
    common[0] = 0x01 | 0x02; // valid_flag0: compatible vibration + haptics select
    common[1] = 0x00; // valid_flag1: leave lightbar / player LEDs alone
    common[2] = small; // motor_right (weak / high-frequency)
    common[3] = large; // motor_left (strong / low-frequency)

    if !bluetooth {
        let mut report = vec![0u8; 48];
        report[0] = 0x02;
        report[1..48].copy_from_slice(&common);
        return report;
    }

    let mut report = vec![0u8; 78];
    report[0] = 0x31;
    report[1] = (bt_seq & 0x0F) << 4;
    report[2] = 0x10;
    report[3..50].copy_from_slice(&common);
    let crc = output_crc32(0xA2, &report[..74]);
    report[74..78].copy_from_slice(&crc.to_le_bytes());
    report
}

#[cfg(windows)]
mod imp {
    use super::{
        ControllerState, SONY_VID, SonyControllerKind, build_output_report,
        parse_dualshock4_report, parse_report, sony_controller_kind,
    };
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA,
        SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW,
        SetupDiGetDeviceInterfaceDetailW,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid,
    };
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile, WriteFile,
    };
    use windows_sys::core::GUID;

    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;

    /// Latest native Sony-pad snapshot, or `None` when none is connected.
    pub type Shared = Arc<Mutex<Option<ControllerState>>>;

    /// The connected device as the writer thread needs it: its path (for a
    /// second, write-side handle) and its transport (USB vs Bluetooth output
    /// framing). `generation` bumps on every reconnect so the writer knows a
    /// fresh pad starts with silent motors.
    #[derive(Clone)]
    struct Route {
        path: Vec<u16>,
        bluetooth: bool,
        generation: u64,
    }

    type SharedRoute = Arc<Mutex<Option<Route>>>;

    /// Host → controller rumble target. `large<<8 | small`, both `0..=255`;
    /// the writer thread only touches the device when this differs from what
    /// the connected pad already has, so callers may set it every frame.
    #[derive(Clone)]
    pub struct RumbleTx(Arc<AtomicU32>);

    impl RumbleTx {
        pub fn set(&self, large: u8, small: u8) {
            self.0
                .store(((large as u32) << 8) | small as u32, Ordering::Relaxed);
        }
    }

    /// Native Sony-pad input plus the existing DualSense rumble target.
    /// DualShock 4 output reports are deliberately not guessed yet.
    pub struct DualSense {
        pub input: Shared,
        rumble: RumbleTx,
    }

    impl DualSense {
        /// Set the desired motor state; a no-op (beyond an atomic store) when
        /// no DualSense is connected or the value is unchanged. A connected
        /// DualShock 4 receives input only, so this remains a no-op for it.
        pub fn set_rumble(&self, large: u8, small: u8) {
            self.rumble.set(large, small);
        }
    }

    /// Spawn the background reader + rumble-writer threads and return their
    /// shared endpoints. The threads hot-plug-retry and never need joining.
    #[must_use]
    pub fn spawn() -> DualSense {
        let shared: Shared = Arc::new(Mutex::new(None));
        let route: SharedRoute = Arc::new(Mutex::new(None));
        let rumble = RumbleTx(Arc::new(AtomicU32::new(0)));
        let reader_shared = shared.clone();
        let reader_route = route.clone();
        let _ = std::thread::Builder::new()
            .name("sony-hid-reader".into())
            .spawn(move || read_loop(&reader_shared, &reader_route));
        let writer_target = rumble.0.clone();
        let _ = std::thread::Builder::new()
            .name("sony-hid-writer".into())
            .spawn(move || write_loop(&writer_target, &route));
        DualSense {
            input: shared,
            rumble,
        }
    }

    fn read_loop(shared: &Shared, route: &SharedRoute) {
        let mut generation = 0u64;
        loop {
            if let Some((handle, path, kind)) = open_sony_controller() {
                let controller = match kind {
                    SonyControllerKind::DualSense => "DualSense",
                    SonyControllerKind::DualShock4 => "DualShock 4",
                };
                tracing::info!(controller, "Sony controller connected (raw HID)");
                generation += 1;
                read_device(handle, shared, route, path, kind, generation);
                tracing::info!(controller, "Sony controller disconnected");
            }
            *route.lock().unwrap() = None;
            *shared.lock().unwrap() = None;
            std::thread::sleep(Duration::from_millis(1000));
        }
    }

    /// Read reports from an open device until it errors/unplugs; closes the
    /// handle before returning. The first parsed report identifies the
    /// transport (`0x31` = Bluetooth), which publishes the device to the
    /// rumble writer thread.
    fn read_device(
        handle: HANDLE,
        shared: &Shared,
        route: &SharedRoute,
        path: Vec<u16>,
        kind: SonyControllerKind,
        generation: u64,
    ) {
        // Bluetooth quirk: requesting feature report 0x05 switches the pad into
        // the full 0x31 input report. Harmless (and ignored) over USB.
        if kind == SonyControllerKind::DualSense {
            let mut feature = [0u8; 41];
            feature[0] = 0x05;
            // SAFETY: `handle` is a live HID handle; `feature` is a valid
            // buffer of the passed length. Result is intentionally ignored.
            unsafe {
                HidD_GetFeature(handle, feature.as_mut_ptr().cast(), feature.len() as u32);
            }
        }

        let mut buffer = [0u8; 256];
        let mut logged = false;
        loop {
            let mut read: u32 = 0;
            // SAFETY: `handle` is live; `buffer`/`read` are valid out-params;
            // synchronous read (null OVERLAPPED).
            let ok = unsafe {
                ReadFile(
                    handle,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as u32,
                    &mut read,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 || read == 0 {
                break;
            }
            let parsed = match kind {
                SonyControllerKind::DualSense => parse_report(&buffer[..read as usize]),
                SonyControllerKind::DualShock4 => parse_dualshock4_report(&buffer[..read as usize]),
            };
            // Diagnostic (once per connect): kind + report id + length + parse
            // result distinguish the DS5 0x01/0x31 and DS4 0x01/0x11 layouts.
            // If it never parses, no input reaches the guest.
            if !logged {
                logged = true;
                tracing::info!(
                    ?kind,
                    report_id = buffer[0],
                    len = read,
                    parsed = parsed.is_some(),
                    "Sony controller first HID input report"
                );
            }
            if let Some(state) = parsed {
                // First parsed report: the transport is now known, so the
                // DualSense rumble writer can build correctly-framed output
                // reports. DS4 input is supported independently; publishing a
                // DS5 output shape to it would be unsafe, so it gets no route.
                if kind == SonyControllerKind::DualSense && route.lock().unwrap().is_none() {
                    let bluetooth = buffer[0] == 0x31;
                    tracing::info!(bluetooth, "DualSense rumble output route ready");
                    *route.lock().unwrap() = Some(Route {
                        path: path.clone(),
                        bluetooth,
                        generation,
                    });
                }
                *shared.lock().unwrap() = Some(state);
            }
        }

        // SAFETY: `handle` came from CreateFileW and is unused hereafter.
        unsafe {
            CloseHandle(handle);
        }
    }

    /// The rumble writer thread: watches the target motor word and the reader
    /// thread's device route, and sends an output report over its own handle
    /// whenever the connected pad's motors differ from the target. A fresh
    /// route generation is assumed silent, so a reconnect re-applies a live
    /// vibration and an idle pad costs zero writes. Write failures drop the
    /// cached handle and retry on the next tick (the reader owns
    /// disconnect/reconnect detection).
    fn write_loop(target: &AtomicU32, route: &SharedRoute) {
        let mut handle = INVALID_HANDLE_VALUE;
        let mut open_generation = 0u64;
        let mut written: u32 = 0;
        let mut bt_seq: u8 = 0;
        loop {
            std::thread::sleep(Duration::from_millis(10));
            let Some(current) = route.lock().unwrap().clone() else {
                if handle != INVALID_HANDLE_VALUE {
                    // SAFETY: `handle` came from CreateFileW and is dropped here.
                    unsafe { CloseHandle(handle) };
                    handle = INVALID_HANDLE_VALUE;
                }
                continue;
            };
            if current.generation != open_generation {
                if handle != INVALID_HANDLE_VALUE {
                    // SAFETY: as above — the old device's handle is stale.
                    unsafe { CloseHandle(handle) };
                    handle = INVALID_HANDLE_VALUE;
                }
                open_generation = current.generation;
                written = 0; // a freshly connected pad has silent motors
            }
            let want = target.load(Ordering::Relaxed);
            if want == written {
                continue;
            }
            if handle == INVALID_HANDLE_VALUE {
                handle = create_file(&current.path, GENERIC_READ | GENERIC_WRITE);
                if handle == INVALID_HANDLE_VALUE {
                    handle = create_file(&current.path, GENERIC_WRITE);
                }
                if handle == INVALID_HANDLE_VALUE {
                    continue; // device busy/gone: retry next tick
                }
            }
            let report =
                build_output_report(current.bluetooth, bt_seq, (want >> 8) as u8, want as u8);
            if current.bluetooth {
                bt_seq = (bt_seq + 1) & 0x0F;
            }
            let mut sent: u32 = 0;
            // SAFETY: `handle` is a live HID handle owned by this thread;
            // `report` is a valid buffer of the passed length; synchronous
            // write (null OVERLAPPED).
            let ok = unsafe {
                WriteFile(
                    handle,
                    report.as_ptr(),
                    report.len() as u32,
                    &mut sent,
                    std::ptr::null_mut(),
                )
            };
            if ok != 0 && sent as usize == report.len() {
                written = want;
            } else {
                // SAFETY: the failed handle is closed exactly once here.
                unsafe { CloseHandle(handle) };
                handle = INVALID_HANDLE_VALUE;
            }
        }
    }

    /// Enumerate present HID interfaces and open a supported Sony controller,
    /// preferring DualSense over DualShock 4 regardless of SetupDi ordering.
    /// A write-capable handle is preferred for the DualSense rumble route;
    /// input-only access is sufficient for either controller generation.
    fn open_sony_controller() -> Option<(HANDLE, Vec<u16>, SonyControllerKind)> {
        let mut ds4_fallbacks: Vec<Vec<u16>> = Vec::new();
        for path in enumerate_hid_paths() {
            // Probe VID/PID with a query-only (access = 0) handle.
            let probe = create_file(&path, 0);
            if probe == INVALID_HANDLE_VALUE {
                continue;
            }
            let mut attrs: HIDD_ATTRIBUTES = unsafe { std::mem::zeroed() };
            attrs.Size = std::mem::size_of::<HIDD_ATTRIBUTES>() as u32;
            // SAFETY: `probe` is a live handle; `attrs` is a valid out-param.
            let ok = unsafe { HidD_GetAttributes(probe, &mut attrs) };
            // SAFETY: `probe` is closed exactly once here.
            unsafe {
                CloseHandle(probe);
            }
            if ok == 0 || attrs.VendorID != SONY_VID {
                continue;
            }
            let Some(kind) = sony_controller_kind(attrs.ProductID) else {
                continue;
            };

            if kind == SonyControllerKind::DualShock4 {
                ds4_fallbacks.push(path);
                continue;
            }

            let mut handle = create_file(&path, GENERIC_READ | GENERIC_WRITE);
            if handle == INVALID_HANDLE_VALUE {
                handle = create_file(&path, GENERIC_READ);
            }
            if handle != INVALID_HANDLE_VALUE {
                return Some((handle, path, kind));
            }
        }

        for path in ds4_fallbacks {
            let mut handle = create_file(&path, GENERIC_READ | GENERIC_WRITE);
            if handle == INVALID_HANDLE_VALUE {
                handle = create_file(&path, GENERIC_READ);
            }
            if handle != INVALID_HANDLE_VALUE {
                return Some((handle, path, SonyControllerKind::DualShock4));
            }
        }
        None
    }

    /// `CreateFileW` on a NUL-terminated wide path with shared read/write.
    fn create_file(path: &[u16], access: u32) -> HANDLE {
        // SAFETY: `path` is NUL-terminated UTF-16; all other args are the
        // standard constants for opening an existing device.
        unsafe {
            CreateFileW(
                path.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        }
    }

    /// Device paths of all present HID interfaces (each NUL-terminated UTF-16),
    /// via the SetupDi device-interface enumeration.
    fn enumerate_hid_paths() -> Vec<Vec<u16>> {
        let mut paths = Vec::new();

        let mut hid_guid: GUID = unsafe { std::mem::zeroed() };
        // SAFETY: `hid_guid` is a valid out-param.
        unsafe {
            HidD_GetHidGuid(&mut hid_guid);
        }

        // SAFETY: standard call; returns INVALID_HANDLE_VALUE on failure.
        let dev_info = unsafe {
            SetupDiGetClassDevsW(
                &hid_guid,
                std::ptr::null(),
                std::ptr::null_mut(),
                DIGCF_PRESENT | DIGCF_DEVICEINTERFACE,
            )
        };
        // HDEVINFO is an `isize` handle (unlike HANDLE); invalid is -1 or 0.
        if dev_info == -1 || dev_info == 0 {
            return paths;
        }

        let mut iface: SP_DEVICE_INTERFACE_DATA = unsafe { std::mem::zeroed() };
        iface.cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DATA>() as u32;

        let mut index = 0u32;
        loop {
            // SAFETY: `dev_info` is live; `iface` is a valid in/out param.
            let got = unsafe {
                SetupDiEnumDeviceInterfaces(
                    dev_info,
                    std::ptr::null(),
                    &hid_guid,
                    index,
                    &mut iface,
                )
            };
            if got == 0 {
                break;
            }
            index += 1;

            // First call sizes the detail buffer.
            let mut required: u32 = 0;
            // SAFETY: null detail buffer + size 0 requests the required size.
            unsafe {
                SetupDiGetDeviceInterfaceDetailW(
                    dev_info,
                    &iface,
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                    std::ptr::null_mut(),
                );
            }
            if required == 0 {
                continue;
            }

            // SP_DEVICE_INTERFACE_DETAIL_DATA_W: cbSize (u32) at [0], then the
            // wide path. cbSize is the documented magic 8 on 64-bit (struct
            // size, NOT the buffer length); the path starts at byte offset 4.
            let mut detail = vec![0u8; required as usize];
            detail[0..4].copy_from_slice(&8u32.to_le_bytes());
            // SAFETY: `detail` is `required` bytes, matching the size argument.
            let ok = unsafe {
                SetupDiGetDeviceInterfaceDetailW(
                    dev_info,
                    &iface,
                    detail.as_mut_ptr().cast(),
                    required,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                continue;
            }
            if let Some(path) = wide_from_bytes(&detail[4..]) {
                paths.push(path);
            }
        }

        // SAFETY: `dev_info` is closed exactly once here.
        unsafe {
            SetupDiDestroyDeviceInfoList(dev_info);
        }
        paths
    }

    /// Read a NUL-terminated UTF-16 string out of a little-endian byte slice,
    /// re-appending the NUL so the result is a valid `CreateFileW` argument.
    fn wide_from_bytes(bytes: &[u8]) -> Option<Vec<u16>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 1 < bytes.len() {
            let unit = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
            if unit == 0 {
                break;
            }
            out.push(unit);
            i += 2;
        }
        if out.is_empty() {
            return None;
        }
        out.push(0);
        Some(out)
    }
}

#[cfg(windows)]
pub use imp::{DualSense, RumbleTx, Shared, spawn};

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 64-byte USB (0x01) input report with the given payload bytes at
    /// their documented offsets (payload starts at index 1).
    // One positional arg per report byte keeps call sites aligned with the
    // documented report layout; a params struct would obscure that mapping.
    #[allow(clippy::too_many_arguments)]
    fn usb_report(
        lx: u8,
        ly: u8,
        rx: u8,
        ry: u8,
        l2: u8,
        r2: u8,
        buttons0: u8,
        buttons1: u8,
        buttons2: u8,
    ) -> Vec<u8> {
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1] = lx;
        r[2] = ly;
        r[3] = rx;
        r[4] = ry;
        r[5] = l2;
        r[6] = r2;
        // r[7] = seq (ignored)
        r[8] = buttons0;
        r[9] = buttons1;
        r[10] = buttons2;
        r
    }

    #[test]
    fn rejects_unknown_or_short_reports() {
        assert!(parse_report(&[]).is_none());
        assert!(
            parse_report(&[0x02, 0, 0, 0]).is_none(),
            "unknown report id"
        );
        assert!(parse_report(&[0x01, 0, 0]).is_none(), "too short for 0x01");
        assert!(
            parse_report(&[0x31, 0, 0, 0, 0]).is_none(),
            "too short for 0x31"
        );
    }

    #[test]
    fn face_buttons_and_hat_decode() {
        // buttons0: Square|Cross|Circle|Triangle set; hat = 2 (East → Right).
        let report = usb_report(128, 128, 128, 128, 0, 0, 0xF0 | 0x02, 0x00, 0x00);
        let s = parse_report(&report).expect("valid USB report");
        assert!(s.square && s.cross && s.circle && s.triangle);
        assert!(s.dpad_right && !s.dpad_up && !s.dpad_down && !s.dpad_left);
    }

    #[test]
    fn shoulder_menu_and_system_buttons_decode() {
        // buttons1: L1|R1|Create|Options|L3|R3; buttons2: PS|TouchPad.
        let report = usb_report(
            128,
            128,
            128,
            128,
            0,
            0,
            0x08, // hat = SE would set dpad; use 0x08 -> hat 8 = centered
            0x01 | 0x02 | 0x10 | 0x20 | 0x40 | 0x80,
            0x01 | 0x02,
        );
        let s = parse_report(&report).expect("valid USB report");
        assert!(s.l1 && s.r1, "L1/R1");
        assert!(s.create, "Create/Share");
        assert!(s.options, "Options");
        assert!(s.l3 && s.r3, "thumb clicks");
        assert!(s.ps_button, "PS/Home");
        assert!(s.touchpad_click, "touchpad click");
        // hat nibble 8 = centered.
        assert!(!s.dpad_up && !s.dpad_down && !s.dpad_left && !s.dpad_right);
    }

    #[test]
    fn sticks_and_triggers_scale() {
        let report = usb_report(0, 255, 128, 128, 255, 0, 0x08, 0, 0);
        let s = parse_report(&report).expect("valid USB report");
        assert!(
            (s.left_stick_x + 1.0).abs() < 1e-2,
            "byte 0 -> -1 (left/up)"
        );
        assert!((s.left_stick_y - 1.0).abs() < 1e-2, "byte 255 -> +1 (down)");
        assert_eq!(s.right_stick_x, 0.0, "byte 128 -> center");
        assert_eq!(s.l2_trigger, 1.0, "full L2");
        assert_eq!(s.r2_trigger, 0.0, "released R2");
        // Centered sticks round-trip to the Orbis center byte.
        let neutral = usb_report(128, 128, 128, 128, 0, 0, 0x08, 0, 0);
        let d = parse_report(&neutral).unwrap().to_orbis_pad_data();
        assert_eq!(d[4], 128);
        assert_eq!(d[5], 128);
    }

    /// USB rumble output report: id 0x02, 48 bytes, valid_flag0 selects
    /// compatible vibration + haptics, motors at payload offsets 2 (small/
    /// right) and 3 (large/left), lightbar untouched.
    #[test]
    fn usb_output_report_carries_motors_at_documented_offsets() {
        let r = build_output_report(false, 0, 200, 55);
        assert_eq!(r.len(), 48);
        assert_eq!(r[0], 0x02, "USB output report id");
        assert_eq!(r[1], 0x03, "valid_flag0: compatible vibration + haptics");
        assert_eq!(r[2], 0x00, "valid_flag1: lightbar/LEDs untouched");
        assert_eq!(r[3], 55, "motor_right = small motor");
        assert_eq!(r[4], 200, "motor_left = large motor");
        assert!(r[5..].iter().all(|&b| b == 0), "rest of the payload silent");
    }

    /// Bluetooth rumble output report: id 0x31, rolling sequence tag in the
    /// high nibble of byte 1, data tag 0x10, same payload shifted to offset
    /// 3, and a trailing little-endian CRC-32 over 0xA2 + bytes 0..74 —
    /// without which the pad silently discards the frame.
    #[test]
    fn bluetooth_output_report_is_sequenced_and_crc_terminated() {
        let r = build_output_report(true, 5, 10, 20);
        assert_eq!(r.len(), 78);
        assert_eq!(r[0], 0x31, "BT output report id");
        assert_eq!(r[1], 5 << 4, "sequence tag in the high nibble");
        assert_eq!(r[2], 0x10, "data tag");
        assert_eq!(r[3], 0x03, "valid_flag0 at the shifted payload offset");
        assert_eq!(r[5], 20, "motor_right = small motor");
        assert_eq!(r[6], 10, "motor_left = large motor");
        let crc = u32::from_le_bytes(r[74..78].try_into().unwrap());
        assert_eq!(crc, output_crc32(0xA2, &r[..74]), "trailing CRC-32");
        // The sequence tag wraps within its nibble.
        assert_eq!(build_output_report(true, 0x1F, 0, 0)[1], 0xF0);
    }

    /// The CRC is the standard reflected CRC-32: check against a known
    /// vector ("123456789" -> 0xCBF43926) by feeding the seed as the first
    /// data byte.
    #[test]
    fn output_crc32_matches_the_standard_crc32_check_vector() {
        let data = b"123456789";
        assert_eq!(output_crc32(data[0], &data[1..]), 0xCBF4_3926);
    }

    #[test]
    fn bluetooth_report_parses_at_shifted_offset() {
        // 0x31 payload is shifted one byte further (seq at [1], payload at [2]).
        let mut r = vec![0u8; 78];
        r[0] = 0x31;
        r[2] = 128; // lx
        r[3] = 128; // ly
        r[4] = 128; // rx
        r[5] = 128; // ry
        r[9] = 0x08; // buttons0: hat centered
        r[10] = 0x20; // buttons1: Options
        let s = parse_report(&r).expect("valid BT report");
        assert!(s.options, "BT Options at shifted offset");
        assert_eq!(s.left_stick_x, 0.0, "BT centered stick");
    }

    #[test]
    fn dualshock4_usb_report_maps_native_playstation_controls() {
        let mut report = vec![0u8; 64];
        report[0] = 0x01;
        report[1] = 0; // left X
        report[2] = 255; // left Y
        report[3] = 128; // right X
        report[4] = 128; // right Y
        report[5] = 0xF0 | 0x02; // all face buttons + D-pad right
        report[6] = 0x01 | 0x02 | 0x10 | 0x20 | 0x40 | 0x80;
        report[7] = 0x01 | 0x02; // PS + touchpad click
        report[8] = 255; // analog L2
        report[9] = 64; // analog R2

        let state = parse_dualshock4_report(&report).expect("valid DS4 USB report");
        assert!(state.square && state.cross && state.circle && state.triangle);
        assert!(state.dpad_right && !state.dpad_up && !state.dpad_down && !state.dpad_left);
        assert!(state.l1 && state.r1 && state.create && state.options);
        assert!(state.l3 && state.r3 && state.ps_button && state.touchpad_click);
        assert!((state.left_stick_x + 1.0).abs() < 1e-2);
        assert!((state.left_stick_y - 1.0).abs() < 1e-2);
        assert_eq!(state.l2_trigger, 1.0);
        assert!((state.r2_trigger - 64.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn dualshock4_bluetooth_report_uses_the_extended_payload_offset() {
        let mut report = vec![0u8; 78];
        report[0] = 0x11;
        report[3] = 128; // left X
        report[4] = 128; // left Y
        report[5] = 128; // right X
        report[6] = 128; // right Y
        report[7] = 0x08; // centered hat
        report[8] = 0x20; // Options
        report[9] = 0x01; // PS
        report[10] = 32; // L2
        report[11] = 224; // R2

        let state = parse_dualshock4_report(&report).expect("valid DS4 Bluetooth report");
        assert!(state.options && state.ps_button);
        assert_eq!(state.left_stick_x, 0.0);
        assert!((state.l2_trigger - 32.0 / 255.0).abs() < 1e-6);
        assert!((state.r2_trigger - 224.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn dualshock4_minimal_bluetooth_report_uses_the_common_payload() {
        let mut report = vec![0u8; 10];
        report[0] = 0x01;
        report[1] = 128; // left X
        report[2] = 128; // left Y
        report[3] = 128; // right X
        report[4] = 128; // right Y
        report[5] = 0x20 | 0x08; // Cross + centered hat
        report[6] = 0x10; // Share/Create
        report[7] = 0x02; // touchpad click
        report[8] = 16; // L2
        report[9] = 240; // R2

        let state = parse_dualshock4_report(&report).expect("valid minimal DS4 BT report");
        assert!(state.cross && state.create && state.touchpad_click);
        assert_eq!(state.left_stick_x, 0.0);
        assert!((state.l2_trigger - 16.0 / 255.0).abs() < 1e-6);
        assert!((state.r2_trigger - 240.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn sony_product_ids_classify_both_controller_generations() {
        assert_eq!(
            sony_controller_kind(DUALSENSE_PID),
            Some(SonyControllerKind::DualSense)
        );
        assert_eq!(
            sony_controller_kind(DUALSENSE_EDGE_PID),
            Some(SonyControllerKind::DualSense)
        );
        assert_eq!(
            sony_controller_kind(DUALSHOCK4_V1_PID),
            Some(SonyControllerKind::DualShock4)
        );
        assert_eq!(
            sony_controller_kind(DUALSHOCK4_V2_PID),
            Some(SonyControllerKind::DualShock4)
        );
        assert_eq!(sony_controller_kind(0x1234), None);
    }
}
