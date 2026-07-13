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
}

impl Default for HleRegistry {
    fn default() -> Self {
        Self::new()
    }
}
