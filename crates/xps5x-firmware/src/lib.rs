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
    DecryptedSelf, KeyProvider, KeyRequest, NoKeysProvider, SegmentKey, decrypt_self, require_key,
};
pub use dynlib::linker::{
    HLE_TRAMPOLINE_BASE, HleTrampoline, LinkedModule, UNRESOLVED_STUB_ADDR, link_module,
};
pub use pup::Firmware;
pub use registry::{ModulePolicy, ModuleRegistry, Resolver};
pub use report::summarize;
pub use slb2::{Slb2Entry, parse_slb2};
pub use sprx::{SprxModule, SprxSegment, TlsTemplate, parse_sprx, proc_param_sdk_version};

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

    // M1-D (wall #4): surface the NEEDED dependency chain loudly instead of
    // silently dropping it. Imports resolve by NID against the HLE registry
    // regardless of which module declares them, so an HLE-covered NEEDED
    // entry is informational; one with no matching HLE library is the first
    // sign a title needs a real file-backed `.prx` load (future work).
    if !dynlib_data.needed_modules.is_empty() {
        let hle_libs: std::collections::HashSet<String> = hle
            .registered_names()
            .into_iter()
            .map(|(lib, _)| lib)
            .collect();
        for needed in &dynlib_data.needed_modules {
            let stem = needed.trim_end_matches(".sprx").trim_end_matches(".prx");
            if hle_libs.contains(stem) {
                tracing::info!("NEEDED {needed}: covered by HLE library '{stem}'");
            } else {
                tracing::warn!(
                    "NEEDED {needed}: no HLE library named '{stem}' — its imports resolve only if \
                     their NIDs are registered elsewhere (file-backed .prx loading not implemented)"
                );
            }
        }
    }

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
