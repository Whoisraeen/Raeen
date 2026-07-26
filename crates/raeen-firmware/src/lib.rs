//! # Raeen Firmware
//!
//! The "firmware spine": ingests PS5 firmware packages (PUP/SLB2),
//! decrypts SELF/module payloads through a **user-supplied** [`KeyProvider`]
//! (Raeen ships no keys), and — in later milestones — parses and links
//! Sony's real `.sprx` modules by NID against HLE or LLE implementations.
//!
//! This crate never contains or extracts Sony keys or firmware. See the
//! design spec, section 2, for the clean-room boundary.

/// Crate identifier, used in diagnostics.
pub const CRATE_NAME: &str = "raeen-firmware";

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
    HLE_TRAMPOLINE_BASE, HleTrampoline, LinkedModule, ModuleInit, ModuleInitRole, ProcessTables,
    UNRESOLVED_STUB_BASE, UnresolvedImport, UnresolvedStub, link_module, link_module_into,
};
pub use pup::Firmware;
pub use registry::{ModulePolicy, ModuleRegistry, Resolver};
pub use report::summarize;
pub use slb2::{Slb2Entry, parse_slb2};
pub use sprx::{
    SprxModule, SprxSegment, StaticTlsModule, TlsTemplate, UnwindInfo, parse_sprx,
    proc_param_sdk_version, static_tls_total,
};

use raeen_core::error::FirmwareError;

/// The decoded static view of a module: its parsed `.sprx` structure plus the
/// decoded dynamic tables (imports, exports, relocations, NEEDED names).
#[derive(Debug, Clone)]
pub struct InspectedModule {
    /// The parsed module (segments, entry, TLS template, …).
    pub module: sprx::SprxModule,
    /// The decoded dynamic tables: imports, exports, relocations, NEEDED
    /// dependency names, and the import library/module name maps.
    pub dynlib: dynlib::DynlibData,
}

/// SELF decrypt-or-passthrough -> `.sprx` parse -> dynamic-table decode, with
/// no relocation and no HLE: the static view of what a module imports,
/// exports, and needs. [`load_module`] is built on this, so a diagnostics
/// tool and the loader can never disagree about a module's contents.
///
/// Errors propagate exactly as in [`load_module`]: an encrypted SELF with no
/// matching key is a genuine, propagated `Err` (a caller with no key gets
/// `FirmwareError::MissingKey`, not a partial result); a module with no
/// `dynamic`/`dynlib_data` decodes as zero imports/exports/relocations.
pub fn inspect_module(
    bytes: &[u8],
    provider: &dyn crypto::KeyProvider,
) -> Result<InspectedModule, FirmwareError> {
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
    let dynlib = match &standard {
        Some((image, tags)) => dynlib::parse_dynlibdata(image, tags)?,
        None => dynlib::parse_dynlibdata(module.dynlib_data.as_deref().unwrap_or(&[]), &dyn_tags)?,
    };
    Ok(InspectedModule { module, dynlib })
}

/// End-to-end LM1 pipeline: [`inspect_module`] (SELF decrypt-or-passthrough ->
/// `.sprx` parse -> `PT_SCE_DYNLIBDATA` decode) -> export registration -> link.
///
/// The decoded exports are registered into `registry` (so later-loaded modules
/// can resolve LLE imports against this one), then [`dynlib::linker::link_module`]
/// performs the actual relocation. An unresolved import NID is recorded in the
/// returned [`LinkedModule::unresolved`] and logged, non-fatal — only a genuine
/// parse/decrypt/link error propagates as `Err`.
pub fn load_module(
    bytes: &[u8],
    provider: &dyn crypto::KeyProvider,
    registry: &mut registry::ModuleRegistry,
    hle: &raeen_hle::HleRegistry,
    base: u64,
) -> Result<dynlib::linker::LinkedModule, FirmwareError> {
    let InspectedModule {
        module,
        dynlib: dynlib_data,
    } = inspect_module(bytes, provider)?;

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
    /// `image_offset`. Feed this to `raeen_runtime::GuestArena`.
    pub linked: dynlib::linker::LinkedModule,
    /// The dependencies that were file-loaded, in load order.
    pub dependencies: Vec<LoadedDependency>,
}

fn append_main_initializer(
    inits: &mut Vec<dynlib::linker::ModuleInit>,
    module_name: &str,
    init_vaddr: Option<u64>,
) {
    let Some(init_vaddr) = init_vaddr else {
        return;
    };
    let name = if module_name.is_empty() {
        "main executable".to_string()
    } else {
        module_name.to_string()
    };
    tracing::info!("{name}: main module_start (DT_INIT) at +{init_vaddr:#x} before process entry");
    inits.push(dynlib::linker::ModuleInit {
        name,
        image_offset: init_vaddr,
        role: dynlib::linker::ModuleInitRole::Main,
    });
}

/// Schedule a file-backed dependency's `module_start` (DT_INIT) at
/// `image_offset` into the composed process image.
///
/// Tagged [`Dependency`](dynlib::linker::ModuleInitRole::Dependency): a
/// dependency has no crt0 that re-runs it, so the runtime runs it under **every**
/// entry policy. Mislabeling it `Main` would make a process entry
/// (`CrtOwnsMainInit`) silently withhold it and never run the dependency's
/// constructors — the exact init-ordering regression this role distinction
/// exists to prevent. Symmetric with [`append_main_initializer`]; the unit test
/// pins the role at this single production site.
fn append_dependency_initializer(
    inits: &mut Vec<dynlib::linker::ModuleInit>,
    name: &str,
    image_offset: u64,
) {
    inits.push(dynlib::linker::ModuleInit {
        name: name.to_string(),
        image_offset,
        role: dynlib::linker::ModuleInitRole::Dependency,
    });
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

/// How deep under the app directory the `.prx` index walk descends.
///
/// Titles nest their modules only a couple of levels (`Media/Modules`,
/// `Media/Plugins`), so 4 is generous; the bound exists so a pathological
/// tree — or a save/content directory full of unrelated files — cannot turn
/// process load into an unbounded filesystem walk.
const MODULE_SCAN_MAX_DEPTH: usize = 4;

/// Directories the index walk never descends into, matched case-insensitively
/// against a single path component.
///
/// `sce_sys/` is package metadata (icons, `param.json`, `pic0.png`) and never
/// holds modules. The rest are bulk content directories where a recursive walk
/// buys nothing but IO.
const MODULE_SCAN_SKIP_DIRS: &[&str] = &["sce_sys", "savedata", "streamingassets"];

/// App-relative directories whose `.prx` files initialize **before** `_start`
/// even when no `DT_NEEDED` entry names them, matched case-insensitively
/// against the directory path relative to the app root.
///
/// This is the standard Unity-on-PS5 layout: `Media/Modules` holds modules the
/// title expects already started (its IL2CPP assemblies and platform shim),
/// while `Media/Plugins` holds native plugins it activates itself through
/// `sceKernelLoadStartModule`. SharpEmu's loader classifies the same two
/// directories the same way (`SharpEmuRuntime.cs:636-645`, `StartAtBoot`).
const EAGER_PLUGIN_DIRS: &[&str] = &["media/modules"];

/// Bounds on the transitive `DT_NEEDED` walk (see [`load_process`]).
///
/// The walk is a fixpoint over a visit-set, so cycles and diamonds terminate
/// on their own; these bounds are the safety net for a pathological or
/// hostile module graph (a NEEDED chain that names a fresh module forever),
/// not something a real title should ever touch — the measured retail eboot
/// names 50 direct modules. A module that would cross either bound is cut,
/// LOUDLY, one warning per module, and its imports stay unresolved — exactly
/// like a missing file. The total bound only gates *transitively discovered*
/// modules: direct NEEDEDs and scanned plugins are all queued before the
/// walk starts and can never be cut by it.
const MAX_DEPENDENCY_DEPTH: usize = 8;
const MAX_LOADED_MODULES: usize = 64;

/// Size of the guest arena's image region (`raeen_runtime::arena::IMAGE_SIZE`),
/// mirrored here so the process loader can name an over-budget composition
/// itself rather than letting it surface as an opaque map failure.
///
/// `raeen-firmware` does not depend on `raeen-runtime` (the dependency runs the
/// other way), so this is a checked duplicate, pinned by
/// `composed_image_budget_matches_the_guest_arena_image_region`.
pub const GUEST_IMAGE_REGION_BYTES: u64 = 0x4000_0000; // 1 GiB

/// A module decoded far enough to link: SELF -> ELF -> `.sprx` -> dynlib data.
struct DecodedModule {
    module: sprx::SprxModule,
    dynlib: dynlib::DynlibData,
}

/// A dependency decoded and placed in pass 1, awaiting linking in pass 2.
struct PendingDep {
    name: String,
    /// Main-module `DT_NEEDED` dependencies initialize before `_start`;
    /// optional app plugins are merely placed and await LoadStartModule.
    eager_init: bool,
    /// Image offset within the composed process image.
    offset: u64,
    /// Absolute guest base (`process base + offset`).
    base: u64,
    decoded: DecodedModule,
}

/// One `.prx` the process loader has been asked to file-load: a main-module
/// `DT_NEEDED`, a root-level plugin from the directory scan, or — discovered
/// while loading — another dependency's own `DT_NEEDED` (transitive).
struct ModuleRequest {
    /// The `DT_NEEDED` / file name, e.g. `libfmod.prx`.
    name: String,
    /// Initialize before `_start` (a hard dependency) vs merely pre-place for
    /// a runtime LoadStartModule (an optional plugin). A transitive request
    /// inherits its requirer's: a hard dependency's own NEEDEDs are equally
    /// hard requirements of the process, and a plugin's stay lazy with it.
    eager_init: bool,
    /// `DT_NEEDED` hops from the main module (direct dependencies are 1),
    /// bounded by [`MAX_DEPENDENCY_DEPTH`].
    depth: usize,
    /// Who named this module. The missing-file warning must say whose imports
    /// stay unresolved, and for a transitive miss that is the dependency that
    /// required it, not the eboot.
    required_by: String,
    /// Exact file this request already resolved to, when it came from the
    /// directory index rather than a `DT_NEEDED` name. Carrying it avoids
    /// re-finding the file by basename, which would pick the wrong one when a
    /// title ships two same-named modules in different directories.
    path: Option<std::path::PathBuf>,
}

/// A per-process stack-protector canary: derived from
/// [`std::collections::hash_map::RandomState`]'s per-process random keys (no
/// new dependency), low byte forced to zero (glibc's "terminator canary"
/// convention, so string functions can't leak it) and guaranteed nonzero — a
/// zero canary would let stack-protected code "work" against zeroed memory
/// rather than proving a real install.
///
/// Deliberately mirrors `raeen_runtime`'s `fs:0x28` canary rather than sharing
/// it: the two are independent ABIs (see [`build_hle_data_page`]), and the
/// runtime's is private to that crate.
fn stack_canary() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(0x5A_FE_57_AC_C4_AA_2D_01);
    let masked = hasher.finish() & !0xFF;
    if masked == 0 { 0x100 } else { masked }
}

/// Build the process's **HLE data page** and register its symbols as LLE
/// exports at their absolute guest addresses.
///
/// # Why a page, and why here
///
/// Some imports are *data*, not functions. The HLE registry can only say "this
/// NID is a function" — it hands out a trampoline address, which is a *code*
/// marker the runtime traps. A data symbol needs the opposite: a real, readable
/// guest address holding a real value. Resolving one as a trampoline (or
/// leaving it unresolved) means the guest dereferences a marker address and
/// faults, which is exactly how the measured title died: `libc.prx`'s
/// stack-protector prologue read libkernel's `__stack_chk_guard` global, found
/// the unresolved stub in the slot, and faulted dereferencing it.
///
/// So this reserves a page *inside the guest image* — plain guest memory the
/// arena already maps — writes the values into it, and registers each symbol as
/// an ordinary LLE export at `page_base + offset`. No new runtime mechanism,
/// no new mapped region: to the linker these are just exports that happen to
/// come from us instead of from a `.prx`.
///
/// It must run **before any module is linked**, since a linker can only resolve
/// what is already registered — which is why the page is reserved at a known
/// offset up front rather than appended afterwards.
///
/// `__stack_chk_guard` is independent of the runtime's `fs:0x28` canary: they
/// are two different stack-protector ABIs (global-variable vs TCB-slot), and
/// compiled code reads the same one in both prologue and epilogue, so the two
/// values need not agree.
fn build_hle_data_page(registry: &mut registry::ModuleRegistry, page_base: u64) -> Vec<u8> {
    let mut page: Vec<u8> = Vec::new();
    let mut exports: Vec<dynlib::SymbolExport> = Vec::new();

    let add = |name: &str, bytes: &[u8], page: &mut Vec<u8>, exports: &mut Vec<_>| {
        // 8-byte align every entry: these are word-sized globals.
        while !page.len().is_multiple_of(8) {
            page.push(0);
        }
        let offset = page.len() as u64;
        page.extend_from_slice(bytes);
        exports.push(dynlib::SymbolExport {
            nid: dynlib::nid::nid_of(name),
            value: offset,
        });
        tracing::debug!("HLE data export {name} at {:#x}", page_base + offset);
    };

    add(
        "__stack_chk_guard",
        &stack_canary().to_le_bytes(),
        &mut page,
        &mut exports,
    );
    // Standard IPv6 address constants exported by libSceNet. These are data,
    // not functions: native guest code dereferences their resolved addresses.
    add("in6addr_any", &[0u8; 16], &mut page, &mut exports);
    let mut loopback = [0u8; 16];
    loopback[15] = 1;
    add("in6addr_loopback", &loopback, &mut page, &mut exports);

    // `__progname` is libkernel's `char *` global naming the running program
    // (BSD convention; libc uses it in error/abort messages). It is a POINTER
    // export, so two entries live in the page: the string bytes, then the
    // exported 8-byte slot holding their absolute guest address.
    while !page.len().is_multiple_of(8) {
        page.push(0);
    }
    let progname_offset = page.len() as u64;
    page.extend_from_slice(b"eboot.bin\0");
    add(
        "__progname",
        &(page_base + progname_offset).to_le_bytes(),
        &mut page,
        &mut exports,
    );

    // `Need_sceLibcInternal` is libSceLibcInternal's exported `int` flag: a
    // nonzero value tells the title's own libc/CRT glue that the internal
    // libc is present and should be used. It is DATA — the guest reads it,
    // never calls it — so resolving it as an HLE trampoline would hand the
    // reader code-marker bytes; a page slot holding 1 is the honest value
    // (measured: ASTRO.BOT imports it from libSceLibcInternal).
    add(
        "Need_sceLibcInternal",
        &1u32.to_le_bytes(),
        &mut page,
        &mut exports,
    );

    // Most of this page is libkernel's, so it is registered there. But
    // `resolve` is **provider-aware** — it keys on the library the importing
    // symbol names, not the NID alone — so a constant must be registered under
    // every provider a title actually imports it from. The IPv6 constants are
    // libSceNet's (see their `add` above), and the measured Minecraft imports
    // `in6addr_any` naming `libSceNet`; registering them only under `libkernel`
    // left that import unresolved and stopped the title's boot.
    let net_nids = [
        dynlib::nid::nid_of("in6addr_any"),
        dynlib::nid::nid_of("in6addr_loopback"),
    ];
    let net_exports: Vec<dynlib::SymbolExport> = exports
        .iter()
        .filter(|export| net_nids.contains(&export.nid))
        .map(|export| dynlib::SymbolExport {
            nid: export.nid,
            value: export.value,
        })
        .collect();
    let libc_internal_nids = [dynlib::nid::nid_of("Need_sceLibcInternal")];
    let libc_internal_exports: Vec<dynlib::SymbolExport> = exports
        .iter()
        .filter(|export| libc_internal_nids.contains(&export.nid))
        .map(|export| dynlib::SymbolExport {
            nid: export.nid,
            value: export.value,
        })
        .collect();
    registry.register_module_exports_at("libkernel", &exports, page_base);
    registry.register_module_exports_at("libSceNet", &net_exports, page_base);
    registry.register_module_exports_at("libSceLibcInternal", &libc_internal_exports, page_base);
    tracing::info!(
        "HLE data page: {} symbol(s), {:#x} bytes at {page_base:#x}",
        exports.len(),
        page.len()
    );
    page
}

/// The `(provider, symbol)` data exports [`build_hle_data_page`] registers, in
/// static form for tools that model import resolution without building a
/// process image (e.g. `cargo xtask nids coverage`). Kept in sync by
/// `tests::hle_data_page_resolves_every_listed_export`.
pub fn hle_data_page_export_names() -> &'static [(&'static str, &'static str)] {
    &[
        ("libkernel", "__stack_chk_guard"),
        ("libkernel", "in6addr_any"),
        ("libkernel", "in6addr_loopback"),
        ("libkernel", "__progname"),
        ("libkernel", "Need_sceLibcInternal"),
        ("libSceNet", "in6addr_any"),
        ("libSceNet", "in6addr_loopback"),
        ("libSceLibcInternal", "Need_sceLibcInternal"),
    ]
}

/// SELF decrypt-or-passthrough -> `parse_sprx` -> dynamic decode, handling both
/// dynamic models (a `PT_SCE_DYNLIBDATA` blob, or a real title's standard
/// vaddr-based tags). Shared by the main module and each dependency.
fn decrypt_and_decode(
    bytes: &[u8],
    provider: &dyn crypto::KeyProvider,
) -> Result<DecodedModule, FirmwareError> {
    let decrypted = crypto::self_crypto::decrypt_self(bytes, provider)?;
    let module = sprx::parse_sprx(&decrypted.elf)?;
    let dyn_tags = match &module.dynamic {
        Some(d) => dynlib::parse_sce_dynamic(d)?,
        None => Vec::new(),
    };
    let standard = dynlib::standard_dynamic_view(&module.segments, &dyn_tags);
    let dynlib = match &standard {
        Some((image, tags)) => dynlib::parse_dynlibdata(image, tags)?,
        None => dynlib::parse_dynlibdata(module.dynlib_data.as_deref().unwrap_or(&[]), &dyn_tags)?,
    };
    Ok(DecodedModule { module, dynlib })
}

/// One `.prx`/`.sprx` found by the app-directory index walk.
struct IndexedModule {
    /// File name as it sits on disk, e.g. `libfmod.prx`.
    name: String,
    /// Absolute path to the file.
    path: std::path::PathBuf,
    /// Lowercased directory path relative to the app root, `/`-separated and
    /// empty for the app root itself, e.g. `media/plugins`. Used both for
    /// eager/lazy classification and for stable ordering.
    rel_dir: String,
}

/// Every `.prx`/`.sprx` shipped anywhere under the app directory, indexed once
/// per process load.
///
/// Titles do not keep all their modules in one place. The system modules a
/// title overrides live in `sce_module/`, but engine-owned modules live
/// wherever the engine puts them — for Unity that is `Media/Modules` (IL2CPP
/// assemblies, platform shim) and `Media/Plugins` (native plugins). Searching
/// only the app root and `sce_module/` made every one of those a missing file:
/// the module was never placed, its exports never registered, and the guest's
/// later `sceKernelLoadStartModule` had nothing to find.
struct ModuleIndex {
    entries: Vec<IndexedModule>,
}

impl ModuleIndex {
    /// Walk `dir` to [`MODULE_SCAN_MAX_DEPTH`], collecting every `.prx`/`.sprx`.
    ///
    /// Symlinks are not followed (a link loop would otherwise defeat the depth
    /// bound), [`MODULE_SCAN_SKIP_DIRS`] are pruned, and an unreadable
    /// directory is skipped rather than failing the load — a title with an
    /// unreadable content directory should still boot.
    fn build(dir: &std::path::Path) -> Self {
        let mut entries = Vec::new();
        Self::walk(dir, dir, 0, &mut entries);
        // Stable, shallowest-first order so pre-placement offsets and log
        // output do not depend on filesystem enumeration order.
        entries.sort_by(|a, b| {
            let depth = |e: &IndexedModule| {
                e.rel_dir.matches('/').count() + usize::from(!e.rel_dir.is_empty())
            };
            depth(a)
                .cmp(&depth(b))
                .then_with(|| a.rel_dir.cmp(&b.rel_dir))
                .then_with(|| {
                    a.name
                        .to_ascii_lowercase()
                        .cmp(&b.name.to_ascii_lowercase())
                })
        });
        Self { entries }
    }

    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        depth: usize,
        out: &mut Vec<IndexedModule>,
    ) {
        if depth > MODULE_SCAN_MAX_DEPTH {
            return;
        }
        let Ok(read) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in read.flatten() {
            // `file_type` on the entry does not follow symlinks, so a link
            // loop can never be recursed into.
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() {
                let component = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if MODULE_SCAN_SKIP_DIRS.contains(&component.as_str()) {
                    continue;
                }
                Self::walk(root, &path, depth + 1, out);
            } else if file_type.is_file()
                && path.extension().is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("prx") || ext.eq_ignore_ascii_case("sprx")
                })
            {
                let rel_dir = path
                    .parent()
                    .and_then(|parent| parent.strip_prefix(root).ok())
                    .map(|rel| {
                        rel.components()
                            .map(|c| c.as_os_str().to_string_lossy().to_ascii_lowercase())
                            .collect::<Vec<_>>()
                            .join("/")
                    })
                    .unwrap_or_default();
                out.push(IndexedModule {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    path,
                    rel_dir,
                });
            }
        }
    }

    /// The indexed file whose name matches `needed` after canonicalization
    /// (case- and extension-insensitive), shallowest-first.
    ///
    /// Canonical matching is what lets a `DT_NEEDED` of `PS5Util.prx` find a
    /// file named `ps5util.prx`, and a guest path ending `.sprx` find the
    /// `.prx` actually shipped.
    fn find(&self, needed: &str) -> Option<&IndexedModule> {
        let want = registry::canonical_module_name(needed);
        self.entries
            .iter()
            .find(|entry| registry::canonical_module_name(&entry.name) == want)
    }
}

/// Whether an indexed module's directory means "initialize before `_start`".
fn is_eager_plugin_dir(rel_dir: &str) -> bool {
    EAGER_PLUGIN_DIRS.contains(&rel_dir)
}

/// Indices into `pending` ordered so every module's own `DT_NEEDED`
/// dependencies initialize **before** it — a post-order depth-first walk of
/// the NEEDED graph.
///
/// The load walk is breadth-first, which is right for *placing* modules but
/// wrong for *initializing* them: it yields the main module's `DT_NEEDED` list
/// in declaration order. Measured on Subnautica Below Zero, that ran
/// `Il2CppUserAssemblies.prx`'s `module_start` first and its own `libc.prx`
/// third — so IL2CPP's initializer called into a libc whose `module_start` had
/// not run, and died calling a still-null function pointer. A real loader
/// initializes dependencies first, and the guest assumes it.
///
/// Modules not in `pending` (HLE-covered or missing) are simply not edges.
/// A NEEDED cycle cannot be satisfied in any order; it is broken at the
/// back-edge with a warning, leaving the rest of the order intact.
fn topological_init_order(pending: &[PendingDep]) -> Vec<usize> {
    let graph: Vec<(&str, &[String])> = pending
        .iter()
        .map(|p| (p.name.as_str(), p.decoded.dynlib.needed_modules.as_slice()))
        .collect();
    init_order_of(&graph)
}

/// [`topological_init_order`] over a plain `(name, needed)` graph, so the
/// ordering rule can be tested without building ELF fixtures.
fn init_order_of(modules: &[(&str, &[String])]) -> Vec<usize> {
    /// Depth-first visit marks.
    const UNVISITED: u8 = 0;
    const IN_PROGRESS: u8 = 1;
    const DONE: u8 = 2;

    fn visit(
        idx: usize,
        modules: &[(&str, &[String])],
        index_of: &std::collections::HashMap<String, usize>,
        mark: &mut [u8],
        order: &mut Vec<usize>,
    ) {
        if mark[idx] != UNVISITED {
            return;
        }
        mark[idx] = IN_PROGRESS;
        for needed in modules[idx].1 {
            let Some(&dep) = index_of.get(&registry::canonical_module_name(needed)) else {
                continue; // HLE-covered or not shipped — no ordering constraint
            };
            if dep == idx {
                continue;
            }
            if mark[dep] == IN_PROGRESS {
                tracing::warn!(
                    "NEEDED cycle: {} <-> {} — initializing {} first and breaking the cycle",
                    modules[idx].0,
                    modules[dep].0,
                    modules[dep].0
                );
                continue;
            }
            visit(dep, modules, index_of, mark, order);
        }
        mark[idx] = DONE;
        order.push(idx);
    }

    let index_of: std::collections::HashMap<String, usize> = modules
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (registry::canonical_module_name(name), i))
        .collect();
    let mut mark = vec![UNVISITED; modules.len()];
    let mut order = Vec::with_capacity(modules.len());
    // Seed in load order so the result is deterministic and, for a graph with
    // no constraints, identical to the old behaviour.
    for idx in 0..modules.len() {
        visit(idx, modules, &index_of, &mut mark, &mut order);
    }
    order
}

/// Locate a `DT_NEEDED` module's file: `dir/<needed>` first, then each of
/// [`DEPENDENCY_SUBDIRS`], then anywhere the [`ModuleIndex`] walk found it.
/// `None` if it ships nowhere we look.
///
/// The explicit probes come first so the documented precedence (app root, then
/// `sce_module/`) is preserved exactly; the index only ever *adds* reach.
fn find_dependency_file(
    dir: &std::path::Path,
    needed: &str,
    index: &ModuleIndex,
) -> Option<std::path::PathBuf> {
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
    index.find(needed).map(|entry| entry.path.clone())
}

// ---------------------------------------------------------------------------
// Loader symbol-override policies
//
// `ModuleRegistry::force_hle_nid` intercepts ONE symbol of an otherwise-LLE
// module. Two policies use it, with opposite gating, and both live here
// rather than inline in the load loop so the loop reads as policy names:
//
// * the mspace family below is forced HLE **by default** — it encodes a
//   measured, current limitation of the LLE path, not a preference;
// * [`apply_diagnostic_overrides`] is pure diagnostic tooling — every trap
//   there is env-gated and OFF by default.
// ---------------------------------------------------------------------------

/// NID of `__cxa_throw` as the measured title's shipped `libc.prx` exports it
/// (matches `nid_of("__cxa_throw")`).
const CXA_THROW_NID: u64 = 0xbe4b_ae2d_f867_4992;

/// The `sceLibcMspace*` allocator family, forced HLE by
/// [`force_hle_mspace_family`]. NIDs measured from the measured title's own
/// shipped `libc.prx` exports (each matches `nid_of` of its comment name).
const MSPACE_FORCE_HLE_NIDS: &[u64] = &[
    0xfe19_f5b5_c547_ab94, // sceLibcMspaceCreate
    0x5ba4_a255_2882_0ed2, // sceLibcMspaceDestroy
    0x3898_e6fd_0388_1e52, // sceLibcMspaceMalloc
    0x5656_bf67_e797_971a, // sceLibcMspaceFree
    0x2d8a_371a_1225_077f, // sceLibcMspaceCalloc
    0x885d_6240_7cf1_0495, // sceLibcMspaceMemalign
    0xa961_1297_25cc_2371, // sceLibcMspacePosixMemalign
    0x8228_2854_766f_54f1, // sceLibcMspaceRealloc
    0x9639_2a31_c0b8_fe69, // sceLibcMspaceAlignedAlloc
    0xa7a9_6b45_6f3f_30b6, // sceLibcMspaceReallocalign
    0x99f1_dd25_322f_86ea, // sceLibcMspaceMallocStats
    0x934e_232d_7bb7_f887, // sceLibcMspaceMallocStatsFast
    0x7c4a_16e8_126c_3ede, // sceLibcMspaceMallocUsableSize
    0xa735_1aec_a128_c9dc, // sceLibcMspaceIsHeapEmpty
];

/// Whether the mspace force-HLE policy is active, given the raw value of
/// `RAEEN_FORCE_HLE_MSPACE` (`None` = unset).
///
/// **Default-off, explicit opt-in.** A shipped libc is `PreferLle` because its
/// stateful allocator family must stay inside one implementation. Measured on
/// ASTRO.BOT (2026-07-21), forcing the family to HLE caused 287,716 main-thread
/// malloc calls to consume 11.3 seconds and prevented the title from reaching
/// Resident Load after 140 seconds. Leaving the family LLE reached Resident
/// Load and the first flip in about 22 seconds in the same build.
///
/// `RAEEN_FORCE_HLE_MSPACE=1` keeps the old workaround available for titles
/// whose shipped allocator cannot initialize against the guest arena. Unset,
/// `=0`, and other values preserve the shipped module's coherent allocator.
///
/// Pure — the env value arrives as an argument — so the decision is testable
/// without mutating process-global state.
fn mspace_force_hle_enabled(env_value: Option<&str>) -> bool {
    matches!(env_value, Some("1"))
}

/// Force every [`MSPACE_FORCE_HLE_NIDS`] symbol of `module` to resolve HLE,
/// even with a shipped module registered `PreferLle`. Applied per loaded
/// module: a module that exports no mspace symbol simply never matches a key,
/// so this is a no-op for it. Separated from the env read so the policy
/// itself is unit-testable.
fn force_hle_mspace_family(registry: &mut registry::ModuleRegistry, module: &str) {
    tracing::debug!("mspace force-HLE diagnostic override applied to {module}");
    for &nid in MSPACE_FORCE_HLE_NIDS {
        registry.force_hle_nid(module, nid);
    }
}

/// Diagnostic symbol overrides — **every one env-gated and off by default**,
/// pure troubleshooting tooling with no place in the normal load path:
///
/// * `RAEEN_TRAP_CXA_THROW` — force-route `__cxa_throw` ([`CXA_THROW_NID`])
///   to the HLE trap so the C++ exception a title's worker threads throw gets
///   NAMED before they die (the exception is uncaught anyway). Everything
///   else in the shipped libc still runs its real code.
fn apply_diagnostic_overrides(registry: &mut registry::ModuleRegistry, module: &str) {
    if std::env::var_os("RAEEN_TRAP_CXA_THROW").is_some() {
        tracing::info!("diagnostic override: trapping __cxa_throw in {module}");
        registry.force_hle_nid(module, CXA_THROW_NID);
    }
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
/// are registered at their **absolute** address.
///
/// # Two passes, and why it must be two
///
/// **Every** module's exports are registered (pass 1) before **any** module is
/// linked (pass 2). A linker can only resolve what is already registered, so a
/// single interleaved pass silently makes resolution depend on `DT_NEEDED`
/// order — and the dependency graph is not a list. Measured on the retail
/// title, whose order is `libcohtml, libRenoirCore, libfmod, libc`: libcohtml
/// is linked first but imports heavily from libc, which is loaded last, so its
/// imports were stubbed for no reason but ordering. Re-linking once every
/// export exists drops libcohtml 886 -> 47 unresolved, libRenoirCore 108 -> 2,
/// libfmod 85 -> 49 — roughly 980 slots that held an unresolved-stub address
/// where a real export belonged. Those are live wrong pointers in guest memory,
/// not a reporting artifact.
///
/// A `DT_NEEDED` that is HLE-covered is deliberately **not** file-loaded (the
/// HLE implementation is preferred and is what `libc`/`libkernel`/`libSce*`
/// resolve through); one that is neither HLE-covered nor present as a file is
/// logged loudly and left unresolved, never silently dropped.
///
/// # Transitive closure
///
/// A direct dependency can import from a module only *it* names: `depA`'s own
/// `DT_NEEDED` list. Loading only the eboot's direct NEEDEDs leaves those
/// imports unresolved no matter how well pass 2 works — the exporting module
/// was never read. So loading is a breadth-first walk over a request queue:
/// each file-loaded dependency contributes its own `needed_modules` as new
/// requests, discovered from the same search dirs, until the closure is
/// reached. The two-pass architecture is unaffected — pass 1 (this walk)
/// registers every export, pass 2 links.
///
/// A visit-set keyed by canonical module name (the same identity the
/// registry resolves providers by) makes the walk a fixpoint: a diamond (`A`
/// and `B` both need `C`) loads `C` once, a cycle (`A` needs `A`) terminates.
/// The set is seeded with the direct NEEDEDs and the scanned root-level
/// plugins, so a plugin a dependency also names keeps its existing
/// pre-placed-not-initialized treatment rather than being reclassified as an
/// eager dependency. The walk is bounded ([`MAX_DEPENDENCY_DEPTH`] hops,
/// [`MAX_LOADED_MODULES`] modules) with a loud warning per cut module, and a
/// missing transitive file degrades exactly like a missing direct one: warn,
/// name the requiring dependency, leave those imports unresolved.
///
/// Transitive dependencies initialize in discovery (BFS) order, after their
/// requirers — the same non-topological order direct dependencies already
/// initialized in (DT_NEEDED order). If a title is measured to need strict
/// reverse-topological `module_start` ordering, that is a separate change to
/// the `module_inits` schedule, not to this walk.
pub fn load_process(
    bytes: &[u8],
    dir: &std::path::Path,
    provider: &dyn crypto::KeyProvider,
    registry: &mut registry::ModuleRegistry,
    hle: &raeen_hle::HleRegistry,
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

    // Index every `.prx`/`.sprx` under the app directory once. Both the
    // `DT_NEEDED` search and the optional-plugin pre-placement scan below read
    // it, so a title that keeps its modules in engine-specific subdirectories
    // (Unity's `Media/Modules`, `Media/Plugins`) is served by the same walk.
    let module_index = ModuleIndex::build(dir);
    tracing::debug!(
        "app module index: {} .prx/.sprx under {}",
        module_index.entries.len(),
        dir.display()
    );

    let mut next_offset = align_up_16k(dynlib::linker::image_size(&module)? as u64);
    let mut dependencies = Vec::new();
    let mut dep_images: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut dep_unwind_modules = Vec::new();
    // Decoded and placed in pass 1, linked in pass 2 (see "Two passes" above).
    let mut pending: Vec<PendingDep> = Vec::new();

    // ONE set of marker tables for the whole process. Every module's HLE
    // trampolines and unresolved stubs are allocated from these, so an index
    // means the same thing everywhere — see `ProcessTables`. Linking each
    // module with private tables (as this used to, via `load_module`) made
    // every module restart at index 0, so a dependency's import #k resolved to
    // the MAIN module's #k at runtime: wrong function, no diagnostic.
    let mut tables = dynlib::linker::ProcessTables::new();

    // The dependency walk's request queue and visit-set. The set is keyed by
    // canonical module name — exactly the identity the registry resolves
    // providers by — and is what makes the transitive walk a fixpoint: a
    // diamond (`A` and `B` both need `C`) loads `C` once, a cycle (`A` needs
    // `A`) terminates. It is seeded with every direct NEEDED and scanned
    // plugin, so a module can never be queued twice regardless of how many
    // modules name it. (This also dedupes a name repeated in the eboot's own
    // NEEDED list, which the old linear pass would have loaded twice.)
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<ModuleRequest> = std::collections::VecDeque::new();
    let main_display_name = if module.name.is_empty() {
        "main module".to_string()
    } else {
        module.name.clone()
    };
    for name in &dynlib_data.needed_modules {
        if visited.insert(registry::canonical_module_name(name)) {
            queue.push_back(ModuleRequest {
                name: name.clone(),
                eager_init: true,
                depth: 1,
                required_by: main_display_name.clone(),
                path: None,
            });
        }
    }
    // App-owned PRX plugins may be loaded later through
    // `sceKernelLoadStartModule`/`Dlsym` and therefore do not appear in the
    // eboot's DT_NEEDED list. Place every PRX shipped anywhere under the app
    // directory in the process image now, so runtime loading can resolve real
    // exports without mutating executable layout mid-flight.
    //
    // The scan used to cover the app root only. Unity titles keep nothing
    // there: their modules live in `Media/Modules` and `Media/Plugins`, so
    // every one of them was a missing file and the guest's LoadStartModule
    // got a code-less pseudo-handle back.
    for entry in &module_index.entries {
        if !visited.insert(registry::canonical_module_name(&entry.name)) {
            continue;
        }
        let eager_init = is_eager_plugin_dir(&entry.rel_dir);
        let where_ = if entry.rel_dir.is_empty() {
            "app root".to_string()
        } else {
            entry.rel_dir.clone()
        };
        tracing::info!(
            "optional app PRX {} ({where_}): preplacing for runtime LoadStartModule{}",
            entry.name,
            if eager_init {
                ", initializing before _start"
            } else {
                ""
            }
        );
        queue.push_back(ModuleRequest {
            name: entry.name.clone(),
            eager_init,
            depth: 1,
            required_by: format!("plugin scan ({where_})"),
            path: Some(entry.path.clone()),
        });
    }

    // Reserve the HLE data page FIRST: its symbols must be registered before
    // any module links, and its address must be known before it is filled.
    let hle_data_offset = next_offset;
    let hle_data = build_hle_data_page(registry, base.wrapping_add(hle_data_offset));
    if !hle_data.is_empty() {
        next_offset = align_up_16k(next_offset + hle_data.len() as u64);
        dep_images.push((hle_data_offset, hle_data));
    }

    // PASS 1 continues as a breadth-first walk: loading one dependency can
    // discover NEW requests (its own DT_NEEDED list — see "Transitive
    // closure" above), queued behind everything already pending. Exports are
    // still all registered here, before ANY module links in pass 2 below.
    while let Some(request) = queue.pop_front() {
        let needed = request.name.as_str();
        let stem = needed.trim_end_matches(".sprx").trim_end_matches(".prx");
        let resolved = match &request.path {
            Some(path) => Some(path.clone()),
            None => find_dependency_file(dir, needed, &module_index),
        };
        let Some(path) = resolved else {
            if hle_libs.contains(stem) {
                tracing::info!("NEEDED {needed}: no file shipped; covered by HLE library '{stem}'");
            } else {
                tracing::warn!(
                    "NEEDED {needed} (required by {}): no HLE library named '{stem}' and no file \
                     anywhere under {} (searched the app root, sce_module/, and {} indexed \
                     .prx/.sprx) — its imports will not resolve",
                    request.required_by,
                    dir.display(),
                    module_index.entries.len()
                );
            }
            continue;
        };
        // A shipped file is loaded even when an HLE library of the same name
        // exists. The shipped module must own every import attributed to it:
        // stateful APIs cannot safely mix its private runtime state with HLE
        // state on a symbol-by-symbol basis.
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
                 — loading it as the preferred provider for its own imports"
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
        // Parse once, register the exports at their ABSOLUTE address. Linking
        // happens in a SECOND pass, once EVERY module's exports exist — see the
        // "Two passes" section above. (This deliberately does not go through
        // `load_module`: that allocates private marker tables — the aliasing bug
        // above — and registers exports module-relative, which is only correct
        // for a module based at 0, forcing a second parse to fix up.)
        let dep = match decrypt_and_decode(&dep_bytes, provider) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("NEEDED {needed}: failed to decode ({e}) — skipping");
                continue;
            }
        };

        let dep_base = base.wrapping_add(next_offset);
        let image_len = dynlib::linker::image_size(&dep.module)? as u64;
        registry.set_policy(needed, registry::ModulePolicy::PreferLle);
        // Symbol-level overrides for the shipped module (see the "loader
        // symbol-override policies" section above `load_process`): the mspace
        // family is forced HLE by default — the LLE mspace path does not work
        // with our arenas yet — and any env-gated diagnostics apply on top.
        if mspace_force_hle_enabled(std::env::var("RAEEN_FORCE_HLE_MSPACE").ok().as_deref()) {
            force_hle_mspace_family(registry, needed);
        }
        apply_diagnostic_overrides(registry, needed);
        registry.register_module_exports_at(needed, &dep.dynlib.exports, dep_base);
        tracing::info!(
            "NEEDED {needed}: at +{next_offset:#x} ({image_len:#x} bytes), {} export(s) registered",
            dep.dynlib.exports.len()
        );

        // Transitive closure: this dependency's own DT_NEEDED list may name
        // modules nothing else asked for. Queue each newly discovered one —
        // the visit-set seeded above makes cycles and diamonds load once, and
        // the bounds keep a pathological graph from composing forever. A
        // dependency's NEEDEDs inherit its `eager_init`: a hard dependency's
        // own requirements are equally hard, a pre-placed plugin's stay lazy.
        for transitive in &dep.dynlib.needed_modules {
            if !visited.insert(registry::canonical_module_name(transitive)) {
                continue;
            }
            if request.depth >= MAX_DEPENDENCY_DEPTH {
                tracing::warn!(
                    "transitive NEEDED {transitive} (required by {needed}): cut by the dependency \
                     depth bound ({MAX_DEPENDENCY_DEPTH}) — its imports will not resolve"
                );
                continue;
            }
            if pending.len() + queue.len() >= MAX_LOADED_MODULES {
                tracing::warn!(
                    "transitive NEEDED {transitive} (required by {needed}): cut by the process \
                     module bound ({MAX_LOADED_MODULES}) — its imports will not resolve"
                );
                continue;
            }
            tracing::info!("transitive NEEDED {transitive}: required by {needed}");
            queue.push_back(ModuleRequest {
                name: transitive.clone(),
                eager_init: request.eager_init,
                depth: request.depth + 1,
                required_by: request.name.clone(),
                path: None,
            });
        }

        pending.push(PendingDep {
            name: request.name.clone(),
            eager_init: request.eager_init,
            offset: next_offset,
            base: dep_base,
            decoded: dep,
        });
        next_offset = align_up_16k(next_offset + image_len);
    }

    // The plugin scan reaches every `.prx` under the app directory, so a title
    // shipping a large module set can now compose an image the guest arena
    // cannot map. `GuestArena::new` would reject it as a bare `MapFailed`,
    // which says nothing about which modules were placed or how big they are;
    // name the overflow here instead, with the per-module sizes, so the fix is
    // obvious.
    if next_offset > GUEST_IMAGE_REGION_BYTES {
        tracing::error!(
            "composed process image is {next_offset} bytes, over the {GUEST_IMAGE_REGION_BYTES}-byte \
             guest image region — the guest arena will refuse to map it"
        );
        for p in &pending {
            tracing::error!(
                "  placed {} at +{:#x} ({} bytes)",
                p.name,
                p.offset,
                dynlib::linker::image_size(&p.decoded.module).unwrap_or(0)
            );
        }
    }

    // Between the passes: assign every module with a (non-empty) `PT_TLS` its
    // slot in the process-wide static TLS area, BEFORE anything links —
    // `TPOFF64`/`DTPMOD64` values are baked in during pass 2. Variant-II
    // packing: the main module sits directly below the TCB (preserving the
    // single-module layout its `TPOFF64`s always had), each dependency below
    // the previous, aligned to `max(p_align, 16)`. Skipping this assignment is
    // not a lesser mode, it is the measured retail-title TLS corruption: four
    // modules' thread-locals folded onto the eboot's block (see
    // `sprx::StaticTlsModule`).
    let mut tls_layout: Vec<sprx::StaticTlsModule> = Vec::new();
    let mut tls_cursor = 0u64;
    let mut assign_tls = |name: &str, tls: &Option<sprx::TlsTemplate>| {
        let template = tls.as_ref().filter(|t| t.mem_size > 0)?;
        let align = template.align.max(16);
        tls_cursor = (tls_cursor + template.mem_size).div_ceil(align) * align;
        let module_id = tls_layout.len() as u64 + 1;
        tls_layout.push(sprx::StaticTlsModule {
            name: name.to_string(),
            module_id,
            tp_offset: tls_cursor,
            template: template.clone(),
        });
        tracing::info!(
            "static TLS: module {module_id} '{name}' at tp-{tls_cursor:#x} \
             (memsz {:#x}, tdata {:#x})",
            template.mem_size,
            template.data.len()
        );
        Some(dynlib::linker::TlsAssignment {
            module_id,
            tp_offset: tls_cursor,
        })
    };
    // TLS module ids count the modules that HAVE TLS, in load order — the
    // main module first, so when it has a `PT_TLS` it keeps the id 1 its
    // relocations have always resolved to.
    let main_tls_assignment = assign_tls(&module.name, &module.tls);
    let dep_tls_assignments: Vec<Option<dynlib::linker::TlsAssignment>> = pending
        .iter()
        .map(|p| assign_tls(&p.name, &p.decoded.module.tls))
        .collect();

    // PASS 2: every export in the process is now registered, so each module can
    // resolve against all the others regardless of DT_NEEDED order.
    for (p, tls_assignment) in pending.iter().zip(&dep_tls_assignments) {
        let linked = match dynlib::linker::link_module_into(
            &p.decoded.module,
            &p.decoded.dynlib,
            registry,
            hle,
            p.base,
            &mut tables,
            *tls_assignment,
        ) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("NEEDED {}: failed to link ({e}) — skipping", p.name);
                continue;
            }
        };
        tracing::info!(
            "NEEDED {}: linked, {} of its own import(s) unresolved",
            p.name,
            linked.unresolved.len()
        );
        dependencies.push(LoadedDependency {
            name: p.name.clone(),
            image_offset: p.offset,
            exports: p.decoded.dynlib.exports.len(),
            unresolved: linked.unresolved.len(),
        });
        dep_unwind_modules.extend(linked.unwind_modules.into_iter().map(|mut unwind| {
            unwind.name = p.name.clone();
            unwind.image_offset = p.offset.wrapping_add(unwind.image_offset);
            unwind
        }));
        dep_images.push((p.offset, linked.image));
    }

    // Now the main module, with every dependency's exports already registered —
    // and sharing the same `tables`, so its marker indices continue the
    // dependencies' rather than colliding with them.
    registry.register_module_exports_at(&module.name, &dynlib_data.exports, base);
    let mut linked = dynlib::linker::link_module_into(
        &module,
        &dynlib_data,
        registry,
        hle,
        base,
        &mut tables,
        main_tls_assignment,
    )?;

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

    // Install the process-wide marker tables: they cover every module, so the
    // runtime can invert a trampoline/stub address from ANY of them. Without
    // this the composed module would carry only the main module's entries and a
    // dependency's call would land on the wrong table row.
    linked.hle_trampolines = tables.hle_trampolines().to_vec();
    linked.unresolved_stubs = tables.unresolved_stubs().to_vec();
    linked.unwind_modules.extend(dep_unwind_modules);
    // The composed module carries the PROCESS layout, not just its own slot:
    // the runtime sizes every thread's static area from this and copies each
    // module's `.tdata` into place, and `__tls_get_addr` resolves each TLS
    // module id against it.
    linked.tls_layout = tls_layout;

    // Each dependency's `module_start` (DT_INIT), dependencies-first — which
    // is what a real loader does and what the guest assumes, at two levels:
    // every dependency runs before the main module (the eboot's constructors
    // use objects a dependency's constructors were supposed to create), and
    // each dependency runs after the dependencies IT names (see
    // `topological_init_order` for the measured failure when it did not).
    // `module_inits` runs before the process entry; see `dynlib::DT_INIT`.
    for idx in topological_init_order(&pending) {
        let p = &pending[idx];
        if !p.eager_init {
            continue;
        }
        let Some(init_vaddr) = p.decoded.dynlib.init else {
            tracing::debug!("{}: no DT_INIT — nothing to initialize", p.name);
            continue;
        };
        let image_offset = p.offset.wrapping_add(init_vaddr);
        tracing::info!(
            "{}: module_start (DT_INIT) at +{image_offset:#x} (module vaddr {init_vaddr:#x})",
            p.name
        );
        append_dependency_initializer(&mut linked.module_inits, &p.name, image_offset);
    }

    // Schedule the executable's own DT_INIT too, but tag it `Main`: the runtime
    // decides WHO calls it by entry policy (see `raeen_runtime::EntryPolicy`).
    // A real process entry (`execute_process`) enters a genuine crt0 `_start`,
    // which walks the executable's own init array itself — so the runtime
    // WITHHOLDS this Main initializer and lets crt0 run it exactly once. Running
    // it here too double-constructs the title's globals (measured on ASTRO.BOT:
    // a list-adding ctor built a cyclic list `_start`'s later walk hung on at
    // `module+0x7426c00`). A crt0-less direct execution (`execute_linked`) runs
    // it, because nothing else would. This bet assumes retail crt0 runs the
    // init array (the SDK's does); a hypothetical `_start` that does not would
    // leave main-module globals uninitialized — revisit per-title if one is
    // measured that behaves that way.
    append_main_initializer(&mut linked.module_inits, &module.name, dynlib_data.init);

    // Count honestly: `linked.unresolved` is the MAIN module's relocations
    // only, while the stub table is process-wide. Reporting one against the
    // other invites exactly the units confusion that produced the
    // "99.6% is libfmod" figure.
    let dep_unresolved: usize = dependencies.iter().map(|d| d.unresolved).sum();
    tracing::info!(
        "process composed: {} dependenc(ies), {:#x}-byte image, {} HLE trampoline(s) \
         (process-wide); unresolved relocations: {} in the main module + {} across its \
         dependencies = {}, over {} distinct missing NID(s)",
        dependencies.len(),
        linked.image.len(),
        linked.hle_trampolines.len(),
        linked.unresolved.len(),
        dep_unresolved,
        linked.unresolved.len() + dep_unresolved,
        linked.unresolved_stubs.len()
    );
    // Name every missing NID up front (project rule: log name+NID loudly).
    // One line per distinct NID turns "313 missing" into an actionable
    // implement-me list without waiting to fault on each one at runtime.
    for stub in &linked.unresolved_stubs {
        // Prefer the hash-verified name: "missing sceKernelGetGPI" is an
        // implement-me list, "missing 4oXYe9Xmk0Q" is a research project.
        tracing::info!(
            "  missing {} — NID {:#018x} ({}) wanted from library '{}'",
            dynlib::nid_names::describe(stub.nid),
            stub.nid,
            dynlib::nid::encode_nid(stub.nid),
            stub.library.as_deref().unwrap_or("<unknown>"),
        );
    }

    Ok(LoadedProcess {
        linked,
        dependencies,
    })
}

#[cfg(test)]
mod tests {
    use crate::dynlib::nid::{NidDatabase, nid_of};
    use crate::{ModuleRegistry, Resolver};

    #[test]
    fn crate_name_is_set() {
        assert_eq!(super::CRATE_NAME, "raeen-firmware");
    }

    /// A throwaway directory tree, removed on drop.
    struct TempTree(std::path::PathBuf);

    impl TempTree {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEQ: AtomicU32 = AtomicU32::new(0);
            let root = std::env::temp_dir().join(format!(
                "raeen-{tag}-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).expect("create temp tree");
            Self(root)
        }

        /// Create `rel` (with parents) holding one byte — the index only ever
        /// looks at names and extensions, never contents.
        fn touch(&self, rel: &str) -> std::path::PathBuf {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().expect("file has a parent"))
                .expect("create parent dirs");
            std::fs::write(&path, b"\0").expect("write fixture file");
            path
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The regression that made Subnautica Below Zero (Unity/IL2CPP) exit one
    /// second into boot: its modules ship under `Media/Modules` and
    /// `Media/Plugins`, which the old app-root + `sce_module/` search never
    /// looked at, so `Il2CppUserAssemblies.prx` — the whole game's logic —
    /// was never placed and `sceKernelLoadStartModule` returned a code-less
    /// pseudo-handle.
    #[test]
    fn module_index_finds_prx_in_unity_subdirectories() {
        let tree = TempTree::new("modindex");
        tree.touch("Media/Modules/Il2CppUserAssemblies.prx");
        tree.touch("Media/Modules/PS5Util.prx");
        tree.touch("Media/Plugins/libfmod.prx");
        tree.touch("sce_module/libc.prx");
        tree.touch("rootplugin.prx");
        // Non-modules must not be indexed.
        tree.touch("Media/Resources/data.dat");

        let index = super::ModuleIndex::build(tree.path());
        let mut found: Vec<&str> = index.entries.iter().map(|e| e.name.as_str()).collect();
        found.sort_unstable();
        assert_eq!(
            found,
            [
                "Il2CppUserAssemblies.prx",
                "PS5Util.prx",
                "libc.prx",
                "libfmod.prx",
                "rootplugin.prx"
            ],
            "every shipped .prx under the app dir must be indexed"
        );

        // Lookup is canonical: case- and extension-insensitive, so a
        // `DT_NEEDED` or guest path spelled differently still resolves.
        assert_eq!(
            index.find("ps5util.PRX").map(|e| e.name.as_str()),
            Some("PS5Util.prx")
        );
        assert_eq!(
            index
                .find("Il2CppUserAssemblies.sprx")
                .map(|e| e.name.as_str()),
            Some("Il2CppUserAssemblies.prx")
        );
        assert!(index.find("libSceNotShipped.prx").is_none());
    }

    #[test]
    fn module_index_prunes_metadata_dirs_and_bounds_depth() {
        let tree = TempTree::new("modindex-bounds");
        tree.touch("sce_sys/should_not_load.prx");
        tree.touch("savedata/should_not_load.prx");
        tree.touch("Media/Plugins/real.prx");
        // One component deeper than the walk descends.
        let too_deep = "a/b/c/d/e/buried.prx";
        tree.touch(too_deep);

        let index = super::ModuleIndex::build(tree.path());
        let names: Vec<&str> = index.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            ["real.prx"],
            "package metadata, save data, and over-deep paths must be pruned"
        );
    }

    /// Unity expects `Media/Modules` already started at `_start` but activates
    /// `Media/Plugins` itself through `sceKernelLoadStartModule`; SharpEmu's
    /// loader classifies the same two directories the same way.
    #[test]
    fn media_modules_initialize_eagerly_while_other_dirs_stay_lazy() {
        assert!(super::is_eager_plugin_dir("media/modules"));
        assert!(!super::is_eager_plugin_dir("media/plugins"));
        assert!(!super::is_eager_plugin_dir(""), "app root stays lazy");
        assert!(!super::is_eager_plugin_dir("sce_module"));
    }

    fn needed(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_string()).collect()
    }

    /// Subnautica Below Zero's real graph. Load (breadth-first) order is
    /// `Il2CppUserAssemblies`, `PS5Util`, `libc` — the eboot's `DT_NEEDED`
    /// declaration order — but IL2CPP needs both of the others, so running its
    /// `module_start` first called into an uninitialized libc and faulted on a
    /// null function pointer. Initialization must invert that.
    #[test]
    fn dependencies_initialize_before_the_modules_that_need_them() {
        let il2cpp = needed(&["PS5Util.prx", "libc.prx", "libkernel"]);
        let ps5util = needed(&["libkernel", "libc.prx"]);
        let libc = needed(&["libkernel"]);
        let modules: Vec<(&str, &[String])> = vec![
            ("Il2CppUserAssemblies.prx", &il2cpp),
            ("PS5Util.prx", &ps5util),
            ("libc.prx", &libc),
        ];

        let order = super::init_order_of(&modules);
        let position = |name: &str| {
            order
                .iter()
                .position(|&i| modules[i].0 == name)
                .expect("every module is scheduled exactly once")
        };
        assert_eq!(order.len(), modules.len(), "no module may be dropped");
        assert!(
            position("libc.prx") < position("PS5Util.prx"),
            "libc must initialize before the module that needs it"
        );
        assert!(
            position("PS5Util.prx") < position("Il2CppUserAssemblies.prx"),
            "PS5Util must initialize before IL2CPP"
        );
        assert!(
            position("libc.prx") < position("Il2CppUserAssemblies.prx"),
            "libc must initialize before IL2CPP"
        );
    }

    /// `libkernel` is HLE-covered and never in `pending`; naming it must not
    /// constrain or drop anything. With no real edges the order is unchanged
    /// from the load order, so titles that were working cannot be reordered.
    #[test]
    fn unshipped_dependencies_impose_no_ordering_and_drop_nothing() {
        let a = needed(&["libkernel", "libSceNetCtl"]);
        let b = needed(&["libkernel"]);
        let modules: Vec<(&str, &[String])> = vec![("a.prx", &a), ("b.prx", &b)];
        assert_eq!(super::init_order_of(&modules), vec![0, 1]);
    }

    /// A NEEDED cycle cannot be satisfied in any order. It must be broken
    /// rather than looping forever or dropping a module.
    #[test]
    fn needed_cycles_are_broken_without_dropping_a_module() {
        let a = needed(&["b.prx"]);
        let b = needed(&["a.prx"]);
        let modules: Vec<(&str, &[String])> = vec![("a.prx", &a), ("b.prx", &b)];

        let order = super::init_order_of(&modules);
        assert_eq!(order.len(), 2, "a cycle must not drop a module");
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1], "each module scheduled exactly once");
    }

    /// A module naming itself must not deadlock or duplicate.
    #[test]
    fn self_referential_needed_is_ignored() {
        let a = needed(&["a.prx"]);
        let modules: Vec<(&str, &[String])> = vec![("a.prx", &a)];
        assert_eq!(super::init_order_of(&modules), vec![0]);
    }

    /// The index only ever *adds* reach — the documented app-root-then-
    /// `sce_module/` precedence must still win, so a title that overrides a
    /// system module at the root keeps overriding it.
    #[test]
    fn dependency_search_prefers_app_root_then_sce_module_then_the_index() {
        let tree = TempTree::new("modsearch");
        let root_copy = tree.touch("libc.prx");
        tree.touch("sce_module/libc.prx");
        tree.touch("Media/Plugins/libc.prx");
        let sce_module_only = tree.touch("sce_module/libSceJobManager.prx");
        let index_only = tree.touch("Media/Modules/PS5Util.prx");

        let index = super::ModuleIndex::build(tree.path());
        assert_eq!(
            super::find_dependency_file(tree.path(), "libc.prx", &index),
            Some(root_copy),
            "an app-root module outranks every other copy"
        );
        assert_eq!(
            super::find_dependency_file(tree.path(), "libSceJobManager.prx", &index),
            Some(sce_module_only),
            "sce_module/ is searched before the index"
        );
        assert_eq!(
            super::find_dependency_file(tree.path(), "PS5Util.prx", &index),
            Some(index_only),
            "a module shipped only in a nested directory is now reachable"
        );
        assert_eq!(
            super::find_dependency_file(tree.path(), "libSceNotShipped.prx", &index),
            None
        );
    }

    #[test]
    fn hle_data_page_exports_ipv6_constants_at_the_real_title_nids() {
        let hle = raeen_hle::HleRegistry::new();
        let mut registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
        let base = 0x1000;
        let page = super::build_hle_data_page(&mut registry, base);

        let addr = match registry.resolve(&hle, "libkernel", nid_of("in6addr_any")) {
            Resolver::Lle { addr, .. } => addr,
            other => panic!("in6addr_any must resolve as guest data, got {other:?}"),
        };
        let offset = usize::try_from(addr - base).expect("page-relative address");
        assert_eq!(&page[offset..offset + 16], &[0u8; 16]);

        let loopback_addr = match registry.resolve(&hle, "libkernel", nid_of("in6addr_loopback")) {
            Resolver::Lle { addr, .. } => addr,
            other => panic!("in6addr_loopback must resolve as guest data, got {other:?}"),
        };
        let loopback_offset = usize::try_from(loopback_addr - base).expect("page-relative address");
        let mut expected = [0u8; 16];
        expected[15] = 1;
        assert_eq!(&page[loopback_offset..loopback_offset + 16], &expected);
    }

    /// The static `(provider, symbol)` view [`hle_data_page_export_names`]
    /// must never drift from what the page actually registers: it exists so
    /// diagnostics (NID coverage) model the same resolution the loader does.
    #[test]
    fn hle_data_page_resolves_every_listed_export() {
        let hle = raeen_hle::HleRegistry::new();
        let mut registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
        let _page = super::build_hle_data_page(&mut registry, 0x1000);
        for (provider, name) in super::hle_data_page_export_names() {
            let nid = nid_of(name);
            assert!(
                matches!(
                    registry.resolve_import(&hle, provider, provider, nid),
                    Resolver::Lle { .. }
                ),
                "{provider}::{name} is listed but does not resolve as guest data"
            );
        }
    }

    #[test]
    fn hle_data_page_exports_progname_as_a_pointer_to_the_program_name() {
        let hle = raeen_hle::HleRegistry::new();
        let mut registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
        let base = 0x1000;
        let page = super::build_hle_data_page(&mut registry, base);

        // The real title imports __progname by the NID in nid_names.txt;
        // pin the hash so the export is reachable by that exact identity.
        assert_eq!(nid_of("__progname"), 0x763c_713a_65ba_fdac);

        let addr = match registry.resolve(&hle, "libkernel", nid_of("__progname")) {
            Resolver::Lle { addr, .. } => addr,
            other => panic!("__progname must resolve as guest data, got {other:?}"),
        };
        // The exported slot holds a pointer into the same page…
        let slot = usize::try_from(addr - base).expect("page-relative address");
        let target = u64::from_le_bytes(page[slot..slot + 8].try_into().unwrap());
        assert!(target > base && target < base + page.len() as u64);
        // …and that pointer names the program.
        let str_off = usize::try_from(target - base).expect("page-relative string");
        assert_eq!(&page[str_off..str_off + 10], b"eboot.bin\0");
    }

    #[test]
    fn main_dt_init_is_scheduled_after_dependency_initializers() {
        use crate::dynlib::linker::ModuleInitRole;

        let mut inits = vec![crate::dynlib::linker::ModuleInit {
            name: "dependency.prx".to_string(),
            image_offset: 0x2000,
            role: ModuleInitRole::Dependency,
        }];

        super::append_main_initializer(&mut inits, "game", Some(0x10));

        assert_eq!(inits.len(), 2);
        assert_eq!(inits[0].name, "dependency.prx");
        assert_eq!(inits[0].role, ModuleInitRole::Dependency);
        assert_eq!(inits[1].name, "game");
        assert_eq!(inits[1].image_offset, 0x10);
        assert_eq!(
            inits[1].role,
            ModuleInitRole::Main,
            "the appended executable initializer must carry the Main role so a \
             process entry withholds it for crt0"
        );

        super::append_main_initializer(&mut inits, "game", None);
        assert_eq!(inits.len(), 2, "an ELF without DT_INIT adds no call");
    }

    #[test]
    fn dependency_initializer_is_tagged_dependency_role() {
        use crate::dynlib::linker::ModuleInitRole;

        // Pins the single production role assignment `load_process`'s dependency
        // loop uses (via `append_dependency_initializer`). A regression to `Main`
        // here would make a process entry silently withhold the dependency's
        // constructors under `CrtOwnsMainInit` — with no other test catching it.
        let mut inits = Vec::new();
        super::append_dependency_initializer(&mut inits, "libfoo.prx", 0x1234);

        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].name, "libfoo.prx");
        assert_eq!(inits[0].image_offset, 0x1234);
        assert_eq!(
            inits[0].role,
            ModuleInitRole::Dependency,
            "dependency initializers must be tagged Dependency so the runtime runs them \
             under every entry policy (a mislabel to Main would withhold them from a crt0 entry)"
        );
    }

    #[test]
    fn shipped_libc_mspace_prefers_lle_by_default_and_hle_is_opt_in() {
        use crate::dynlib::SymbolExport;
        use crate::registry::ModulePolicy;

        let hle = raeen_hle::HleRegistry::new();
        let mut registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));

        // One of the 14 policy NIDs. Pin that the measured literal matches
        // the name hash, so the policy list can never drift from the symbols
        // it claims to intercept, and that the HLE side it routes to exists.
        let nid = nid_of("sceLibcMspaceCreate");
        assert_eq!(nid, 0xfe19_f5b5_c547_ab94);
        assert!(super::MSPACE_FORCE_HLE_NIDS.contains(&nid));
        // Same pin for the diagnostic trap's NID, so it can't drift either.
        assert_eq!(nid_of("__cxa_throw"), super::CXA_THROW_NID);

        // A title ships its own libc.prx (an LLE export for the same NID) and
        // `load_process` marks the shipped module PreferLle.
        registry.register_module_exports("libc.prx", &[SymbolExport { nid, value: 0x1234 }]);
        registry.set_policy("libc.prx", ModulePolicy::PreferLle);
        let resolve = |registry: &ModuleRegistry| {
            registry.resolve_import(&hle, "libc.prx", "libSceLibcInternal", nid)
        };

        // A shipped libc owns its stateful allocator by default. The HLE
        // workaround remains an explicit diagnostic opt-in.
        assert!(!super::mspace_force_hle_enabled(None), "default-off");
        assert!(
            !super::mspace_force_hle_enabled(Some("0")),
            "RAEEN_FORCE_HLE_MSPACE=0 keeps the shipped allocator"
        );
        assert!(super::mspace_force_hle_enabled(Some("1")));

        // Opted out (policy never applied): the PreferLle module's own export
        // wins — this is the raw LLE behavior the opt-out restores.
        assert_eq!(resolve(&registry), Resolver::Lle { addr: 0x1234 });

        // Explicit opt-in applied: the NID resolves HLE even though a shipped
        // module exports it and the module is PreferLle.
        super::force_hle_mspace_family(&mut registry, "libc.prx");
        match resolve(&registry) {
            Resolver::Hle { library, function } => {
                assert_eq!(library, "libSceLibcInternal");
                assert_eq!(function, "sceLibcMspaceCreate");
            }
            other => {
                panic!("expected Resolver::Hle under the opt-in mspace policy, got {other:?}")
            }
        }
    }
}
