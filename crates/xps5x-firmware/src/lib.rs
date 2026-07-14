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

pub mod crypto;
pub mod dynlib;
pub mod pup;
pub mod registry;
pub mod report;
pub mod slb2;
pub mod sprx;

pub use crypto::{
    decrypt_self, require_key, DecryptedSelf, KeyProvider, KeyRequest, NoKeysProvider, SegmentKey,
};
pub use dynlib::linker::{
    link_module, HleTrampoline, LinkedModule, HLE_TRAMPOLINE_BASE, UNRESOLVED_STUB_ADDR,
};
pub use pup::Firmware;
pub use registry::{ModulePolicy, ModuleRegistry, Resolver};
pub use report::summarize;
pub use slb2::{parse_slb2, Slb2Entry};
pub use sprx::{parse_sprx, SprxModule, SprxSegment, TlsTemplate};

use xps5x_core::error::FirmwareError;

/// End-to-end LM1 pipeline: SELF decrypt-or-passthrough -> `.sprx` parse ->
/// `PT_SCE_DYNLIBDATA` decode -> export registration -> link.
///
/// This is the convenience chain Task 6 handed off: [`crypto::decrypt_self`]
/// (missing-key is a genuine, propagated `Err` — a caller with no matching
/// key gets `FirmwareError::MissingKey` here, not a partial result), then
/// [`sprx::parse_sprx`], then (if the module has a `PT_DYNAMIC` segment)
/// [`dynlib::parse_sce_dynamic`] + [`dynlib::parse_dynlibdata`] — a module
/// with no `dynamic`/`dynlib_data` is treated as having zero imports/
/// exports/relocations, not an error. The decoded exports are registered
/// into `registry` (so later-loaded modules can resolve LLE imports against
/// this one), then [`dynlib::linker::link_module`] performs the actual
/// relocation. An unresolved import NID is recorded in the returned
/// [`LinkedModule::unresolved`] and logged, non-fatal — only a genuine
/// parse/decrypt/link error propagates as `Err`.
pub fn load_module(
    bytes: &[u8],
    provider: &dyn crypto::KeyProvider,
    registry: &mut registry::ModuleRegistry,
    hle: &xps5x_hle::HleRegistry,
    base: u64,
) -> Result<dynlib::linker::LinkedModule, FirmwareError> {
    let decrypted = crypto::self_crypto::decrypt_self(bytes, provider)?;
    let module = sprx::parse_sprx(&decrypted.elf)?;
    let dyn_tags = match &module.dynamic {
        Some(d) => dynlib::parse_sce_dynamic(d)?,
        None => Vec::new(),
    };
    let dynlib_data =
        dynlib::parse_dynlibdata(module.dynlib_data.as_deref().unwrap_or(&[]), &dyn_tags)?;
    registry.register_module_exports(&module.name, &dynlib_data.exports);
    dynlib::linker::link_module(&module, &dynlib_data, registry, hle, base)
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_set() {
        assert_eq!(super::CRATE_NAME, "xps5x-firmware");
    }
}
