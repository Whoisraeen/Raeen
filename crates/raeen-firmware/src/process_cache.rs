//! Persistent cache for a composed, relocated, trap-patched process image.
//!
//! Cache files are machine-local under the gitignored shader-cache tree. They
//! may contain decrypted user-owned executable bytes and must never be
//! published or committed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::{GUEST_IMAGE_REGION_BYTES, LoadedProcess, ModuleIndex};

const FORMAT_VERSION: u32 = 1;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct CacheKey(String);

impl CacheKey {
    #[cfg(test)]
    fn test(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Serialize, Deserialize)]
struct CacheMetadata<P> {
    format_version: u32,
    key: String,
    hle_data_offset: u64,
    image_len: u64,
    image_sha1: String,
    process: P,
}

pub(crate) struct CacheHit {
    pub process: LoadedProcess,
    pub hle_data_offset: u64,
}

fn update_bytes(hasher: &mut Sha1, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn update_text(hasher: &mut Sha1, text: &str) {
    update_bytes(hasher, text.as_bytes());
}

/// Cache identity: decrypted main ELF content, dependency-file manifest,
/// runtime resolution surface, loader implementation, relevant overrides, and
/// guest base. Dependency contents are represented by stable path + size +
/// mtime, avoiding a full reread of every large PRX on a warm launch while
/// invalidating normal installs/updates atomically.
pub(crate) fn key(
    decrypted_main_elf: &[u8],
    index: &ModuleIndex,
    base: u64,
    hle: &raeen_hle::HleRegistry,
) -> CacheKey {
    let mut hasher = Sha1::new();
    hasher.update(b"RAEEN-LINKED-PROCESS-CACHE");
    hasher.update(FORMAT_VERSION.to_le_bytes());
    hasher.update(base.to_le_bytes());
    update_text(&mut hasher, env!("CARGO_PKG_VERSION"));
    update_bytes(&mut hasher, decrypted_main_elf);

    // Compile-time loader-source fingerprint. A dirty development build cannot
    // accidentally consume an image produced by different relocation, SELF,
    // TLS, or registry logic merely because Cargo.toml's version was unchanged.
    for source in [
        include_bytes!("process_cache.rs").as_slice(),
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("sprx.rs").as_slice(),
        include_bytes!("registry.rs").as_slice(),
        include_bytes!("dynlib/mod.rs").as_slice(),
        include_bytes!("dynlib/linker.rs").as_slice(),
        include_bytes!("crypto/mod.rs").as_slice(),
        include_bytes!("crypto/self_crypto.rs").as_slice(),
    ] {
        update_bytes(&mut hasher, source);
    }

    for entry in &index.entries {
        update_text(&mut hasher, &entry.rel_dir);
        update_text(&mut hasher, &entry.name);
        match std::fs::metadata(&entry.path) {
            Ok(metadata) => {
                hasher.update(metadata.len().to_le_bytes());
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|duration| duration.as_nanos())
                    .unwrap_or(0);
                hasher.update(modified.to_le_bytes());
            }
            Err(_) => hasher.update([0xFF; 24]),
        }
    }

    let mut names = hle.registered_names();
    names.sort_unstable();
    for (library, function) in names {
        update_text(&mut hasher, &library);
        update_text(&mut hasher, &function);
    }
    for variable in ["RAEEN_FORCE_HLE_MSPACE", "RAEEN_TRAP_CXA_THROW"] {
        update_text(&mut hasher, variable);
        update_text(
            &mut hasher,
            &std::env::var(variable).unwrap_or_else(|_| "<unset>".to_string()),
        );
    }

    CacheKey(format!("{:x}", hasher.finalize()))
}

fn enabled() -> bool {
    std::env::var_os("RAEEN_NO_LINK_CACHE").is_none()
}

fn default_root() -> PathBuf {
    std::env::var_os("RAEEN_LINK_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("shader_cache").join("linked-images"))
}

fn paths(root: &Path, key: &CacheKey) -> (PathBuf, PathBuf) {
    (
        root.join(format!("{}.meta.json", key.0)),
        root.join(format!("{}.image.bin", key.0)),
    )
}

pub(crate) fn load(key: &CacheKey) -> Option<CacheHit> {
    if !enabled() {
        return None;
    }
    match load_from(&default_root(), key) {
        Ok(hit) => hit,
        Err(error) => {
            tracing::warn!("linked-image cache read failed; rebuilding: {error}");
            None
        }
    }
}

fn load_from(root: &Path, key: &CacheKey) -> std::io::Result<Option<CacheHit>> {
    let (metadata_path, image_path) = paths(root, key);
    let metadata_stat = match std::fs::metadata(&metadata_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if metadata_stat.len() > MAX_METADATA_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "linked-image cache metadata exceeds safety bound",
        ));
    }
    let metadata_bytes = std::fs::read(&metadata_path)?;
    let mut metadata: CacheMetadata<LoadedProcess> =
        serde_json::from_slice(&metadata_bytes).map_err(std::io::Error::other)?;
    if metadata.format_version != FORMAT_VERSION || metadata.key != key.0 {
        return Ok(None);
    }
    if metadata.image_len > GUEST_IMAGE_REGION_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "linked-image cache image exceeds guest image region",
        ));
    }
    let image_stat = std::fs::metadata(&image_path)?;
    if image_stat.len() != metadata.image_len {
        return Ok(None);
    }
    let image = std::fs::read(&image_path)?;
    if format!("{:x}", Sha1::digest(&image)) != metadata.image_sha1 {
        return Ok(None);
    }
    if !structure_is_valid(&metadata.process, image.len() as u64) {
        return Ok(None);
    }
    metadata.process.linked.image = image;
    Ok(Some(CacheHit {
        process: metadata.process,
        hle_data_offset: metadata.hle_data_offset,
    }))
}

pub(crate) fn store(key: &CacheKey, hle_data_offset: u64, process: &LoadedProcess) {
    if !enabled() {
        return;
    }
    if let Err(error) = store_to(&default_root(), key, hle_data_offset, process) {
        tracing::warn!("linked-image cache write failed; launch remains valid: {error}");
    }
}

fn store_to(
    root: &Path,
    key: &CacheKey,
    hle_data_offset: u64,
    process: &LoadedProcess,
) -> std::io::Result<()> {
    std::fs::create_dir_all(root)?;
    let (metadata_path, image_path) = paths(root, key);
    let metadata = CacheMetadata {
        format_version: FORMAT_VERSION,
        key: key.0.clone(),
        hle_data_offset,
        image_len: process.linked.image.len() as u64,
        image_sha1: format!("{:x}", Sha1::digest(&process.linked.image)),
        process,
    };
    // Image first, metadata last: a crash cannot publish metadata that points
    // at an image which has not yet been completely written.
    std::fs::write(image_path, &process.linked.image)?;
    std::fs::write(
        metadata_path,
        serde_json::to_vec(&metadata).map_err(std::io::Error::other)?,
    )?;
    Ok(())
}

fn structure_is_valid(process: &LoadedProcess, image_len: u64) -> bool {
    process
        .linked
        .executable_ranges
        .iter()
        .all(|(offset, len)| offset.checked_add(*len).is_some_and(|end| end <= image_len))
        && process
            .linked
            .module_inits
            .iter()
            .all(|init| init.image_offset < image_len)
        && process.linked.unwind_modules.iter().all(|module| {
            module
                .image_offset
                .checked_add(module.unwind.image_size)
                .is_some_and(|end| end <= image_len)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::linker::LinkedModule;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEQUENCE: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "raeen-linked-cache-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_process() -> LoadedProcess {
        LoadedProcess {
            linked: LinkedModule {
                image: b"linked-image".to_vec(),
                base: 0x1000,
                executable_ranges: vec![(0, 6)],
                unresolved: Vec::new(),
                unresolved_stubs: Vec::new(),
                module_inits: Vec::new(),
                hle_trampolines: vec![
                    crate::dynlib::linker::HleTrampoline {
                        library: "libkernel".to_string(),
                        function: "sceKernelDlsym".to_string(),
                        addr: crate::HLE_TRAMPOLINE_BASE,
                    },
                    // A dlsym-only reservation. A cache hit returns without
                    // ever calling `load_process`, so the round-trip below is
                    // what proves a restored process still carries the
                    // reservation the guest's `dlsym` needs an address from.
                    crate::dynlib::linker::HleTrampoline {
                        library: "libkernel".to_string(),
                        function: "scriptingGetMem".to_string(),
                        addr: crate::HLE_TRAMPOLINE_BASE + 8,
                    },
                ],
                entry: 1,
                tls: None,
                tls_layout: Vec::new(),
                procparam_offset: None,
                unwind_modules: Vec::new(),
            },
            dependencies: Vec::new(),
            module_exports: Vec::new(),
        }
    }

    #[test]
    fn linked_process_cache_round_trips_metadata_and_raw_image() {
        let temp = TempDir::new();
        let key = CacheKey::test("roundtrip");
        let expected = sample_process();
        store_to(&temp.0, &key, 0x4000, &expected).unwrap();
        let hit = load_from(&temp.0, &key).unwrap().expect("cache hit");
        assert_eq!(hit.hle_data_offset, 0x4000);
        assert_eq!(hit.process, expected);
        // Named explicitly, not just covered by the whole-struct compare: a
        // cache hit never runs `load_process`, so a restored process must
        // already carry the dlsym-only reservation, and the import count must
        // still exclude it.
        assert_eq!(hit.process.linked.hle_trampolines.len(), 2);
        assert_eq!(hit.process.linked.imported_hle_trampoline_count(), 1);
        assert_eq!(hit.process.linked.reserved_hle_trampoline_count(), 1);
    }

    #[test]
    fn corrupt_image_is_a_cache_miss_not_a_loaded_process() {
        let temp = TempDir::new();
        let key = CacheKey::test("corrupt");
        store_to(&temp.0, &key, 0x4000, &sample_process()).unwrap();
        let (_, image_path) = paths(&temp.0, &key);
        std::fs::write(image_path, b"wrong-length").unwrap();
        assert!(load_from(&temp.0, &key).unwrap().is_none());
    }

    #[test]
    fn cache_restore_rebuilds_hle_data_and_every_module_export_without_unwind() {
        use crate::dynlib::nid::{NidDatabase, nid_of};
        use crate::{LoadedModuleExports, ModuleRegistry, Resolver};

        let hle = raeen_hle::HleRegistry::new();
        let base = 0x1000;
        let hle_data_offset = 0x100;
        let mut sizing_registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
        let page = crate::build_hle_data_page(&mut sizing_registry, base + hle_data_offset);

        let mut process = sample_process();
        process.linked.image = vec![0xAA; hle_data_offset as usize + page.len() + 0x100];
        let export_nid = nid_of("cacheOnlyExport");
        process.module_exports.push(LoadedModuleExports {
            name: "libCache.prx".to_string(),
            image_offset: 0x40,
            exports: vec![crate::dynlib::SymbolExport {
                nid: export_nid,
                value: 0x20,
            }],
            prefer_lle: true,
        });
        assert!(
            process.linked.unwind_modules.is_empty(),
            "the regression specifically covers a module with no unwind record"
        );

        let mut restored_registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
        assert!(crate::restore_cached_registry(
            &mut process,
            hle_data_offset,
            &mut restored_registry,
            base,
        ));
        assert_ne!(
            &process.linked.image[hle_data_offset as usize..hle_data_offset as usize + page.len()],
            vec![0xAA; page.len()].as_slice(),
            "the cached per-process HLE page must be refreshed"
        );
        assert_eq!(
            restored_registry.resolve_import(&hle, "libCache.prx", "libCache", export_nid),
            Resolver::Lle { addr: base + 0x60 },
            "warm-cache registry replay must not depend on unwind metadata"
        );
    }
}
