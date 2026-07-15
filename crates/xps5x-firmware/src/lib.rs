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
    HLE_TRAMPOLINE_BASE, HleTrampoline, LinkedModule, UNRESOLVED_STUB_BASE, UnresolvedImport,
    UnresolvedStub, link_module,
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
    // Two dynamic models exist in the wild (see `dynlib::standard_dynamic_view`):
    // homebrew/.sprx put the tables in a `PT_SCE_DYNLIBDATA` blob addressed by
    // `DT_SCE_*` offsets, while real PS5 titles have no such segment and use the
    // standard `DT_STRTAB`/`DT_SYMTAB`/... tags holding **virtual addresses**.
    // Try the standard model first; fall back to the blob.
    let standard = dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
    let dynlib_data = match &standard {
        Some((image, tags)) => dynlib::parse_dynlibdata(image, tags)?,
        None => dynlib::parse_dynlibdata(module.dynlib_data.as_deref().unwrap_or(&[]), &dyn_tags)?,
    };

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

/// One dependency loaded alongside the main module.
#[derive(Debug, Clone)]
pub struct LoadedDependency {
    /// The `DT_NEEDED` name, e.g. `libfmod.prx`.
    pub name: String,
    /// Offset of this module's image within the composed process image.
    pub image_offset: u64,
    /// How many LLE exports it contributed.
    pub exports: usize,
    /// Imports of its own that didn't resolve.
    pub unresolved: usize,
}

/// A whole process: the main module plus its file-backed `.prx` dependencies,
/// composed into one image (M1-D).
#[derive(Debug)]
pub struct LoadedProcess {
    /// The composed image: main module at offset 0, each dependency at its
    /// `image_offset`. Feed this to `xps5x_runtime::GuestArena`.
    pub linked: dynlib::linker::LinkedModule,
    /// The dependencies that were file-loaded, in load order.
    pub dependencies: Vec<LoadedDependency>,
}

/// Round `v` up to the next 16 KiB boundary — dependencies are placed on a
/// generous alignment so no module's image can bleed into the next.
fn align_up_16k(v: u64) -> u64 {
    (v + 0x3FFF) & !0x3FFF
}

/// Subdirectories of the app directory a `DT_NEEDED` module may live in,
/// searched in order after the app directory itself.
///
/// `sce_module/` is where a title ships the **system** modules it wants used
/// in preference to the console's own — `libc.prx`, `libSceNpCppWebApi.prx`.
/// Missing it is expensive: on the measured retail title `sce_module/libc.prx`
/// alone exports 99.4% of the eboot's import relocations.
const DEPENDENCY_SUBDIRS: &[&str] = &["sce_module"];

/// Locate a `DT_NEEDED` module's file: `dir/<needed>` first, then each of
/// [`DEPENDENCY_SUBDIRS`]. `None` if it ships nowhere we look.
fn find_dependency_file(dir: &std::path::Path, needed: &str) -> Option<std::path::PathBuf> {
    let direct = dir.join(needed);
    if direct.is_file() {
        return Some(direct);
    }
    for sub in DEPENDENCY_SUBDIRS {
        let p = dir.join(sub).join(needed);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Load a title as a **process**: the main module plus every `DT_NEEDED`
/// dependency that exists as a real file next to it (M1-D, wall #4).
///
/// # Why this exists
///
/// Some of a real title's imports are satisfied by libraries that ship *inside
/// the game folder* rather than by HLE — a third-party audio engine
/// (`libfmod.prx`), a UI runtime (`libcohtml.Prospero.prx`). Those are the
/// game's own code: they can never be HLE'd and must be loaded. Loading them
/// contributes their exports, which resolve the main module's imports by NID.
///
/// # Scale, honestly
///
/// This is a real but **small** effect. Measured on a retail PS5 title
/// (Minecraft, 876 distinct imports / 87414 import relocations): the bundled
/// `.prx` supply 116 of those relocations — `libfmod` 54 and
/// `libcohtml.Prospero` 62. The overwhelming majority (86883, 99.4%) are
/// `libc`, which is HLE territory.
///
/// An earlier revision of this comment claimed 86852/87222 (99.6%) were
/// `libfmod`. That was wrong: it came from indexing a symbol's `#lib#` id into
/// the *needed-module* table instead of the *import-library* table, which
/// renamed every library (see `dynlib::DT_SCE_NEEDED_MODULE_1`). Do not
/// resurrect that number.
///
/// # How
///
/// Everything is composed into **one** image, so `GuestArena` (which maps a
/// single image) needs no changes: the main module sits at offset 0 and each
/// dependency at a 16 KiB-aligned offset above it. `link_module` already takes
/// a base, so each module is relocated for `base + its offset`, and its exports
/// are registered at their **absolute** address. Dependencies are linked first
/// so their exports exist before the main module resolves against them.
///
/// A `DT_NEEDED` that is HLE-covered is deliberately **not** file-loaded (the
/// HLE implementation is preferred and is what `libc`/`libkernel`/`libSce*`
/// resolve through); one that is neither HLE-covered nor present as a file is
/// logged loudly and left unresolved, never silently dropped.
pub fn load_process(
    bytes: &[u8],
    dir: &std::path::Path,
    provider: &dyn crypto::KeyProvider,
    registry: &mut registry::ModuleRegistry,
    hle: &xps5x_hle::HleRegistry,
    base: u64,
) -> Result<LoadedProcess, FirmwareError> {
    // Parse (not yet link) the main module: we need its NEEDED list and its
    // image size before anything can be placed above it.
    let decrypted = crypto::self_crypto::decrypt_self(bytes, provider)?;
    let module = sprx::parse_sprx(&decrypted.elf)?;
    let dyn_tags = match &module.dynamic {
        Some(d) => dynlib::parse_sce_dynamic(d)?,
        None => Vec::new(),
    };
    let standard = dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
    let dynlib_data = match &standard {
        Some((image, tags)) => dynlib::parse_dynlibdata(image, tags)?,
        None => dynlib::parse_dynlibdata(module.dynlib_data.as_deref().unwrap_or(&[]), &dyn_tags)?,
    };

    let hle_libs: std::collections::HashSet<String> = hle
        .registered_names()
        .into_iter()
        .map(|(lib, _)| lib)
        .collect();

    let mut next_offset = align_up_16k(dynlib::linker::image_size(&module)? as u64);
    let mut dependencies = Vec::new();
    let mut dep_images: Vec<(u64, Vec<u8>)> = Vec::new();

    for needed in &dynlib_data.needed_modules {
        let stem = needed.trim_end_matches(".sprx").trim_end_matches(".prx");
        let Some(path) = find_dependency_file(dir, needed) else {
            if hle_libs.contains(stem) {
                tracing::info!("NEEDED {needed}: no file shipped; covered by HLE library '{stem}'");
            } else {
                tracing::warn!(
                    "NEEDED {needed}: no HLE library named '{stem}' and no file in {} or its \
                     sce_module/ — its imports will not resolve",
                    dir.display()
                );
            }
            continue;
        };
        // A shipped file is loaded even when an HLE library of the same name
        // exists. That is not a conflict: `ModuleRegistry::resolve` defaults to
        // `PreferHle`, so our implementation still wins for every NID we
        // actually implement, and the real module only supplies the rest.
        //
        // This matters enormously and used to be skipped. The measured title
        // ships its own `sce_module/libc.prx`, whose exports cover 86883 of its
        // 87414 import relocations (99.4%) and 260 of the 758 functions it
        // actually calls — including the one it dies on today. Refusing to load
        // it because "libc is HLE-covered" left all of that unresolved while our
        // libc HLE supplied only ~31 relocations' worth.
        if hle_libs.contains(stem) {
            tracing::info!(
                "NEEDED {needed}: HLE library '{stem}' exists AND the title ships the real module \
                 — loading it; HLE still wins per-NID (PreferHle), the file covers the rest"
            );
        }

        let dep_bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "NEEDED {needed}: found at {} but unreadable: {e}",
                    path.display()
                );
                continue;
            }
        };
        let dep_base = base.wrapping_add(next_offset);
        match load_module(&dep_bytes, provider, registry, hle, dep_base) {
            Ok(linked) => {
                // Re-register this dependency's exports at their ABSOLUTE
                // address: `load_module` registers them module-relative, which
                // is only correct for a module based at 0.
                let dep_module = sprx::parse_sprx(
                    &crypto::self_crypto::decrypt_self(&dep_bytes, provider)?.elf,
                )?;
                let dep_tags = match &dep_module.dynamic {
                    Some(d) => dynlib::parse_sce_dynamic(d)?,
                    None => Vec::new(),
                };
                let dep_std = dynlib::standard_dynamic_view(&dep_module.segments, &dep_tags);
                let dep_dyn = match &dep_std {
                    Some((img, tags)) => dynlib::parse_dynlibdata(img, tags)?,
                    None => dynlib::parse_dynlibdata(
                        dep_module.dynlib_data.as_deref().unwrap_or(&[]),
                        &dep_tags,
                    )?,
                };
                registry.register_module_exports_at(needed, &dep_dyn.exports, dep_base);

                let len = linked.image.len() as u64;
                tracing::info!(
                    "NEEDED {needed}: loaded at +{next_offset:#x} ({len:#x} bytes), {} export(s), \
                     {} of its own import(s) unresolved",
                    dep_dyn.exports.len(),
                    linked.unresolved.len()
                );
                dependencies.push(LoadedDependency {
                    name: needed.clone(),
                    image_offset: next_offset,
                    exports: dep_dyn.exports.len(),
                    unresolved: linked.unresolved.len(),
                });
                dep_images.push((next_offset, linked.image));
                next_offset = align_up_16k(next_offset + len);
            }
            Err(e) => tracing::warn!("NEEDED {needed}: failed to load ({e}) — skipping"),
        }
    }

    // Now the main module, with every dependency's exports already registered.
    registry.register_module_exports_at(&module.name, &dynlib_data.exports, base);
    let mut linked = dynlib::linker::link_module(&module, &dynlib_data, registry, hle, base)?;

    // Compose: main module already occupies [0, its image len); splice each
    // dependency in at its offset.
    if let Some(total) = dep_images.iter().map(|(o, i)| o + i.len() as u64).max() {
        let total = usize::try_from(total).map_err(|_| {
            FirmwareError::MalformedSelf("composed process image overflows usize".to_string())
        })?;
        if linked.image.len() < total {
            linked.image.resize(total, 0);
        }
        for (off, img) in dep_images {
            let at = usize::try_from(off).map_err(|_| {
                FirmwareError::MalformedSelf("dependency image offset overflows".to_string())
            })?;
            linked.image[at..at + img.len()].copy_from_slice(&img);
        }
    }

    tracing::info!(
        "process composed: {} dependenc(ies), {:#x}-byte image, {} HLE trampoline(s), {} unresolved",
        dependencies.len(),
        linked.image.len(),
        linked.hle_trampolines.len(),
        linked.unresolved.len()
    );

    Ok(LoadedProcess {
        linked,
        dependencies,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_is_set() {
        assert_eq!(super::CRATE_NAME, "xps5x-firmware");
    }
}
