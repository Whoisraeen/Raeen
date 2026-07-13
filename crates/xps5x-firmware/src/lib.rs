//! # XPS5X Firmware
//!
//! The "firmware spine": ingests PS5 firmware packages (PUP/SLB2),
//! decrypts SELF/module payloads through a **user-supplied** [`KeyProvider`]
//! (XPS5X ships no keys), and — in later milestones — parses and links
//! Sony's real `.sprx` modules by NID against HLE or LLE implementations.
//!
//! This crate never contains or extracts Sony keys or firmware. See the
//! design spec, section 2, for the clean-room boundary.

/// Crate identifier, used in diagnostics.
pub const CRATE_NAME: &str = "xps5x-firmware";

pub mod slb2;
pub mod pup;
pub mod crypto;

pub use slb2::{parse_slb2, Slb2Entry};
pub use pup::Firmware;
pub use crypto::{require_key, KeyProvider, KeyRequest, NoKeysProvider, SegmentKey};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_set() {
        assert_eq!(super::CRATE_NAME, "xps5x-firmware");
    }
}
