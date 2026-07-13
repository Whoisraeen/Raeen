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

/// Access to the guest (emulated PS5) address space from an HLE function.
///
/// Every implementation must be bounds-checked: an out-of-bounds
/// `guest_addr`/length combination returns `false` (touching nothing)
/// rather than panicking or reading/writing outside the guest's actual
/// backing storage. An HLE function handed a wild pointer by buggy or
/// malicious guest code must never be able to turn that into a host OOB
/// access or a panic.
pub trait GuestMemory {
    /// Read `out.len()` bytes starting at `guest_addr` into `out`. Returns
    /// `false` (leaving `out`'s contents unspecified) if the read would
    /// fall outside the guest's mapped memory.
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool;
    /// Write `data` starting at `guest_addr`. Returns `false` (writing
    /// nothing) if the write would fall outside the guest's mapped memory.
    fn write(&self, guest_addr: u64, data: &[u8]) -> bool;
}

/// Everything an HLE function may touch: the emulated kernel (memory,
/// threads, filesystem, ...) and the guest's address space.
///
/// This is the dispatch-context milestone: before it existed, an HLE
/// function was a bare `fn(&[u64]) -> u64` with no way to read/write guest
/// pointers or reach a live [`xps5x_kernel::OrbisKernel`] — every stub was
/// necessarily a no-op that just logged and returned a plausible value. Now
/// every HLE call gets both, so functions like `memcpy`/`strlen`/
/// `sceKernelMapFlexibleMemory` can do the real operation.
pub struct HleContext<'a> {
    /// The live emulated kernel (memory manager, thread manager, VFS, ...).
    pub kernel: &'a xps5x_kernel::OrbisKernel,
    /// The guest's address space, as seen from wherever this call
    /// originated (e.g. the runtime's mapped module image).
    pub mem: &'a dyn GuestMemory,
}

/// HLE function signature: takes a dispatch context and integer arguments,
/// returns a result.
pub type HleFunction = fn(&HleContext, &[u64]) -> u64;

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
        libkernel::register(&registry);
        libc::register(&registry);
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

    /// Look up and call an HLE function, giving it `ctx` (the kernel +
    /// guest memory) alongside its integer arguments.
    pub fn call(&self, ctx: &HleContext, library: &str, function: &str, args: &[u64]) -> Option<u64> {
        let key = format!("{}::{}", library, function);
        if let Some(func) = self.functions.get(&key) {
            debug!("HLE call: {}({:?})", key, args);
            Some(func(ctx, args))
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
/// A tiny in-memory [`GuestMemory`] backed by a `Vec<u8>`, for unit tests
/// that need to exercise real read/write behavior without a runtime.
pub(crate) struct TestMemory(std::cell::RefCell<Vec<u8>>);

#[cfg(test)]
impl TestMemory {
    pub(crate) fn new(size: usize) -> Self {
        Self(std::cell::RefCell::new(vec![0u8; size]))
    }
}

#[cfg(test)]
impl GuestMemory for TestMemory {
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else { return false };
        let buf = self.0.borrow();
        let Some(end) = addr.checked_add(out.len()) else { return false };
        if end > buf.len() {
            return false;
        }
        out.copy_from_slice(&buf[addr..end]);
        true
    }

    fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else { return false };
        let mut buf = self.0.borrow_mut();
        let Some(end) = addr.checked_add(data.len()) else { return false };
        if end > buf.len() {
            return false;
        }
        buf[addr..end].copy_from_slice(data);
        true
    }
}

/// Build an [`HleContext`] over a test kernel and [`TestMemory`]. Defined at
/// the crate root (not inside `mod tests`) so every submodule's own
/// `#[cfg(test)] mod tests` can reach it as `crate::test_ctx` — Rust
/// visibility lets descendant modules see their ancestors' private items.
#[cfg(test)]
pub(crate) fn test_ctx<'a>(kernel: &'a xps5x_kernel::OrbisKernel, mem: &'a TestMemory) -> HleContext<'a> {
    HleContext { kernel, mem }
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
    fn new_registers_substantially_more_than_the_original_three_libraries() {
        let registry = HleRegistry::new();
        // Before this change, `new()` only wired up libSceSysmodule (3
        // functions), libSceVideoOut (5 functions), and libScePad (4
        // functions) — 12 functions total. Broadening libkernel/libc
        // coverage should push this well past that.
        assert!(
            registry.functions.len() > 12,
            "expected substantially more than the original 3-library baseline (12 functions), got {}",
            registry.functions.len()
        );
    }

    #[test]
    fn representative_libkernel_and_libc_functions_are_implemented_and_callable() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = TestMemory::new(0x1000);
        let ctx = test_ctx(&kernel, &mem);
        let samples: &[(&str, &str)] = &[
            ("libkernel", "sceKernelAllocateDirectMemory"),
            ("libkernel", "scePthreadCreate"),
            ("libc", "malloc"),
            ("libc", "memcpy"),
        ];
        for (library, function) in samples {
            assert!(
                registry.is_implemented(library, function),
                "expected {library}::{function} to be implemented"
            );
            let result = registry.call(&ctx, library, function, &[1, 2, 3, 4]);
            assert!(result.is_some(), "{library}::{function} should return a value, not None");
        }
    }

    #[test]
    fn registered_names_reflects_manual_registration() {
        let registry = HleRegistry {
            functions: DashMap::new(),
        };
        fn stub(_ctx: &HleContext, _args: &[u64]) -> u64 {
            0
        }
        registry.register("libFoo", "someFunction", stub);

        let names = registry.registered_names();
        assert_eq!(names, vec![("libFoo".to_string(), "someFunction".to_string())]);
    }
}
