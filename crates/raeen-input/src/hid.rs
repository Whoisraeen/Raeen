//! DualSense controller reader over raw HID (Windows).
//!
//! Clean-room port of SharpEmu's `WindowsDualSenseReader.cs` +
//! `WindowsHidNative.cs` (GPL-2.0-or-later, © SharpEmu Emulator Project).
//! Zero external dependencies: the device is found by `SetupDi*` enumeration
//! filtered on Sony's VID/PID, opened with `CreateFileW`, and read with
//! `ReadFile`; the fixed input report is parsed at documented offsets. Handles
//! both USB (report id `0x01`) and Bluetooth (extended report id `0x31`).
//!
//! Only input is implemented — rumble / lightbar output reports are a
//! deliberately-omitted follow-up (see the module notes in the crate).
//!
//! The pure [`parse_report`] function carries the whole mapping and is
//! unit-tested without a device; the Windows FFI in `imp` only feeds it.

use crate::ControllerState;

/// Sony Corp. USB vendor id.
const SONY_VID: u16 = 0x054C;
/// DualSense (CFI-ZCT1) product id.
const DUALSENSE_PID: u16 = 0x0CE6;
/// DualSense Edge (CFI-ZCT1) product id.
const DUALSENSE_EDGE_PID: u16 = 0x0DF2;

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

    let (dpad_up, dpad_down, dpad_left, dpad_right) = hat_to_dpad(buttons0);

    Some(ControllerState {
        square: buttons0 & 0x10 != 0,
        cross: buttons0 & 0x20 != 0,
        circle: buttons0 & 0x40 != 0,
        triangle: buttons0 & 0x80 != 0,
        l1: buttons1 & 0x01 != 0,
        r1: buttons1 & 0x02 != 0,
        // buttons1 bits 0x04 / 0x08 are the digital L2/R2; we drive L2/R2 from
        // the analog axes below so the whole pipeline is analog-consistent.
        create: buttons1 & 0x10 != 0, // Share / Create (richer than SharpEmu)
        options: buttons1 & 0x20 != 0,
        l3: buttons1 & 0x40 != 0,
        r3: buttons1 & 0x80 != 0,
        ps_button: buttons2 & 0x01 != 0, // PS / Home (richer than SharpEmu)
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
    })
}

#[cfg(windows)]
mod imp {
    use super::{parse_report, ControllerState, DUALSENSE_EDGE_PID, DUALSENSE_PID, SONY_VID};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use windows_sys::core::GUID;
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW,
        SetupDiGetDeviceInterfaceDetailW, DIGCF_DEVICEINTERFACE, DIGCF_PRESENT,
        SP_DEVICE_INTERFACE_DATA,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid, HIDD_ATTRIBUTES,
    };
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, ReadFile, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };

    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;

    /// Latest DualSense snapshot, or `None` when none is connected.
    pub type Shared = Arc<Mutex<Option<ControllerState>>>;

    /// Spawn the background reader thread and return the shared cache. The
    /// thread hot-plug-retries and never needs joining.
    #[must_use]
    pub fn spawn() -> Shared {
        let shared: Shared = Arc::new(Mutex::new(None));
        let worker = shared.clone();
        let _ = std::thread::Builder::new()
            .name("dualsense-hid-reader".into())
            .spawn(move || read_loop(&worker));
        shared
    }

    fn read_loop(shared: &Shared) {
        let mut announced = false;
        loop {
            if let Some(handle) = open_dualsense() {
                if !announced {
                    tracing::info!("DualSense controller connected (raw HID)");
                    announced = true;
                }
                read_device(handle, shared);
                if announced {
                    tracing::info!("DualSense controller disconnected");
                    announced = false;
                }
            }
            *shared.lock().unwrap() = None;
            std::thread::sleep(Duration::from_millis(1000));
        }
    }

    /// Read reports from an open device until it errors/unplugs; closes the
    /// handle before returning.
    fn read_device(handle: HANDLE, shared: &Shared) {
        // Bluetooth quirk: requesting feature report 0x05 switches the pad into
        // the full 0x31 input report. Harmless (and ignored) over USB.
        let mut feature = [0u8; 41];
        feature[0] = 0x05;
        // SAFETY: `handle` is a live HID handle; `feature` is a valid buffer of
        // the passed length. Result is intentionally ignored.
        unsafe {
            HidD_GetFeature(handle, feature.as_mut_ptr().cast(), feature.len() as u32);
        }

        let mut buffer = [0u8; 256];
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
            if let Some(state) = parse_report(&buffer[..read as usize]) {
                *shared.lock().unwrap() = Some(state);
            }
        }

        // SAFETY: `handle` came from CreateFileW and is unused hereafter.
        unsafe {
            CloseHandle(handle);
        }
    }

    /// Enumerate present HID interfaces, returning the first whose VID/PID is a
    /// DualSense; opens it read+write (falling back to read-only).
    fn open_dualsense() -> Option<HANDLE> {
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
            if ok == 0
                || attrs.VendorID != SONY_VID
                || (attrs.ProductID != DUALSENSE_PID && attrs.ProductID != DUALSENSE_EDGE_PID)
            {
                continue;
            }

            let mut handle = create_file(&path, GENERIC_READ | GENERIC_WRITE);
            if handle == INVALID_HANDLE_VALUE {
                handle = create_file(&path, GENERIC_READ);
            }
            if handle != INVALID_HANDLE_VALUE {
                return Some(handle);
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
pub use imp::{spawn, Shared};

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 64-byte USB (0x01) input report with the given payload bytes at
    /// their documented offsets (payload starts at index 1).
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
        assert!(parse_report(&[0x02, 0, 0, 0]).is_none(), "unknown report id");
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
        assert!((s.left_stick_x + 1.0).abs() < 1e-2, "byte 0 -> -1 (left/up)");
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
}
