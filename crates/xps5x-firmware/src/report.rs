//! Human-readable firmware inspection report (drives `--firmware-info`).

use crate::Firmware;
use std::fmt::Write;

/// Render a plaintext summary of a firmware container's entries.
pub fn summarize(firmware: &Firmware) -> String {
    let entries = firmware.entries();
    let mut s = String::new();
    let plural = if entries.len() == 1 { "entry" } else { "entries" };
    let _ = writeln!(s, "SLB2 firmware container: {} {}", entries.len(), plural);
    for (i, e) in entries.iter().enumerate() {
        let _ = writeln!(
            s,
            "  [{i}] {:<20} offset={:#x} size={} bytes (encrypted payload; not decrypted)",
            e.name, e.offset, e.size
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Firmware;

    fn synthetic_firmware() -> Vec<u8> {
        let mut buf = vec![0u8; 0x20 + 0x30];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[0x0C..0x10].copy_from_slice(&1u32.to_le_bytes());
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes());
        buf[0x24..0x28].copy_from_slice(&0x100u32.to_le_bytes());
        buf[0x30..0x30 + 14].copy_from_slice(b"PS5UPDATE1.PUP");
        buf
    }

    #[test]
    fn summary_lists_entries() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        let text = summarize(&fw);
        assert!(text.contains("1 entry"));
        assert!(text.contains("PS5UPDATE1.PUP"));
        assert!(text.contains("0x400")); // offset 2 * 0x200
    }
}
