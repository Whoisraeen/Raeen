//! # XPS5X HLE
//!
//! High-Level Emulation of PS5 system libraries.
//!
//! PS5 games link against Sony's proprietary `.sprx` libraries.
//! Rather than loading encrypted firmware modules, XPS5X re-implements
//! these libraries' exported functions, routing calls to the appropriate
//! emulator subsystem.
//!
//! ## Implemented Libraries
//!
//! | Library | Status | Routes to |
//! |:---|:---|:---|
//! | libkernel.sprx | Partial | xps5x-kernel |
//! | libc.sprx | Partial | Rust std |
//! | libSceGnmDriver.sprx | Stub | xps5x-gpu |
//! | libSceVideoOut.sprx | Stub | xps5x-gpu (Vulkan swapchain) |
//! | libSceAudioOut.sprx | Stub | xps5x-audio |
//! | libScePad.sprx | Stub | xps5x-input |
//! | libSceNet.sprx | Stub | Host networking |
//! | libSceSaveData.sprx | Stub | xps5x-kernel (VFS) |
//! | libSceSysmodule.sprx | Partial | Module registry |

pub mod libc;
pub mod libkernel;
pub mod libsce_gnm_driver;
pub mod libsce_video_out;
pub mod libsce_audio_out;
pub mod libsce_pad;
pub mod libsce_net;
pub mod libsce_save_data;
pub mod libsce_sysmodule;

use dashmap::DashMap;
use tracing::{debug, info, warn};

/// HLE function signature: takes arguments, returns result.
pub type HleFunction = fn(&[u64]) -> u64;

/// Registry of all HLE'd library functions.
pub struct HleRegistry {
    /// Map of "library::function" → implementation.
    functions: DashMap<String, HleFunction>,
}

impl HleRegistry {
    /// Create and populate the HLE registry with all implemented functions.
    pub fn new() -> Self {
        info!("Initializing HLE registry");
        let registry = Self {
            functions: DashMap::new(),
        };

        // Register all implemented HLE functions.
        libsce_sysmodule::register(&registry);
        libsce_video_out::register(&registry);
        libsce_pad::register(&registry);

        info!("HLE registry: {} functions registered", registry.functions.len());
        registry
    }

    /// Register an HLE function.
    pub fn register(&self, library: &str, function: &str, implementation: HleFunction) {
        let key = format!("{}::{}", library, function);
        debug!("HLE register: {}", key);
        self.functions.insert(key, implementation);
    }

    /// Look up and call an HLE function.
    pub fn call(&self, library: &str, function: &str, args: &[u64]) -> Option<u64> {
        let key = format!("{}::{}", library, function);
        if let Some(func) = self.functions.get(&key) {
            debug!("HLE call: {}({:?})", key, args);
            Some(func(args))
        } else {
            warn!("HLE: unimplemented function {}", key);
            None
        }
    }

    /// Check if a function is implemented.
    pub fn is_implemented(&self, library: &str, function: &str) -> bool {
        let key = format!("{}::{}", library, function);
        self.functions.contains_key(&key)
    }

    /// Every registered function as `(library, function)` pairs.
    ///
    /// Each internal key is `"library::function"`; this splits on the first
    /// `"::"` to recover the pair. Used to seed a `NidDatabase` from what the
    /// HLE registry actually implements.
    pub fn registered_names(&self) -> Vec<(String, String)> {
        self.functions
            .iter()
            .filter_map(|entry| {
                entry
                    .key()
                    .split_once("::")
                    .map(|(library, function)| (library.to_string(), function.to_string()))
            })
            .collect()
    }
}

impl Default for HleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_names_splits_library_and_function() {
        let registry = HleRegistry::new();
        let names = registry.registered_names();
        assert!(!names.is_empty(), "HleRegistry::new() should register some functions");
        assert_eq!(names.len(), registry.functions.len());

        // Every name must round-trip: is_implemented(lib, func) is true for
        // each pair we enumerated.
        for (library, function) in &names {
            assert!(
                registry.is_implemented(library, function),
                "registered_names produced ({library}, {function}) that is_implemented doesn't recognize"
            );
        }
    }

    #[test]
    fn registered_names_reflects_manual_registration() {
        let registry = HleRegistry {
            functions: DashMap::new(),
        };
        fn stub(_args: &[u64]) -> u64 {
            0
        }
        registry.register("libFoo", "someFunction", stub);

        let names = registry.registered_names();
        assert_eq!(names, vec![("libFoo".to_string(), "someFunction".to_string())]);
    }
}
