//! Shader cache — stores compiled SPIR-V on disk for reuse.

use dashmap::DashMap;
use std::path::PathBuf;
use tracing::{debug, info};

/// In-memory + on-disk shader cache.
pub struct ShaderCache {
    /// In-memory cache: ISA hash → SPIR-V bytecode.
    memory_cache: DashMap<u64, Vec<u32>>,
    /// Disk cache directory.
    cache_dir: PathBuf,
    /// Cache hit count.
    hits: std::sync::atomic::AtomicU64,
    /// Cache miss count.
    misses: std::sync::atomic::AtomicU64,
}

impl ShaderCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        info!("Shader cache initialized at {}", cache_dir.display());
        Self {
            memory_cache: DashMap::new(),
            cache_dir,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Look up a shader by its ISA hash.
    pub fn get(&self, isa_hash: u64) -> Option<Vec<u32>> {
        // Check memory cache first.
        if let Some(spirv) = self.memory_cache.get(&isa_hash) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            debug!("Shader cache HIT (memory): {:#x}", isa_hash);
            return Some(spirv.clone());
        }

        // Check disk cache.
        let disk_path = self.cache_dir.join(format!("{:016x}.spv", isa_hash));
        if disk_path.exists()
            && let Ok(data) = std::fs::read(&disk_path)
                && data.len() % 4 == 0 {
                    let spirv: Vec<u32> = data
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Promote to memory cache.
                    self.memory_cache.insert(isa_hash, spirv.clone());
                    self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    debug!("Shader cache HIT (disk): {:#x}", isa_hash);
                    return Some(spirv);
                }

        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug!("Shader cache MISS: {:#x}", isa_hash);
        None
    }

    /// Insert a compiled shader into the cache.
    pub fn insert(&self, isa_hash: u64, spirv: Vec<u32>) {
        // Write to disk.
        if let Err(e) = std::fs::create_dir_all(&self.cache_dir) {
            tracing::warn!("Failed to create shader cache dir: {}", e);
        } else {
            let disk_path = self.cache_dir.join(format!("{:016x}.spv", isa_hash));
            let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&disk_path, &bytes) {
                tracing::warn!("Failed to write shader cache: {}", e);
            }
        }

        // Insert into memory cache.
        self.memory_cache.insert(isa_hash, spirv);
        debug!("Shader cached: {:#x}", isa_hash);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (u64, u64) {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        (hits, misses)
    }
}
