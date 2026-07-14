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

pub(crate) mod fmt;
pub mod libc;
pub mod libkernel;
pub mod libsce_app_content;
pub mod libsce_audio_out;
pub mod libsce_common_dialog;
pub mod libsce_gnm_driver;
pub mod libsce_net;
pub mod libsce_np;
pub mod libsce_pad;
pub mod libsce_playgo;
pub mod libsce_save_data;
pub mod libsce_sysmodule;
pub mod libsce_system_service;
pub mod libsce_user_service;
pub mod libsce_video_out;

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

/// Allocates and releases guest memory on behalf of an HLE function —
/// `malloc`/`free`/`realloc`/`mmap`/`munmap`'s underlying mechanism.
///
/// Every method is total: an exhausted arena, an overflowing size/alignment
/// request, or an unrecognized address returns a sentinel (`None`, or simply
/// doing nothing) rather than panicking. Nothing calls this trait's methods
/// yet (that lands in RT2 Task 3/4/5, once a real implementation —
/// `xps5x-runtime`'s `GuestArena` — exists); it is threaded through
/// [`HleContext`] now so every call site is ready ahead of that.
pub trait GuestAllocator {
    /// Allocate at least `size` bytes, aligned to `align`, returning the
    /// guest address of the new block, or `None` if the request cannot be
    /// satisfied (exhausted arena, overflowing size/align, ...).
    fn alloc(&self, size: u64, align: u64) -> Option<u64>;
    /// Release a block previously returned by `alloc`/`realloc`/`mmap`. An
    /// unrecognized `addr` is simply ignored.
    fn free(&self, addr: u64);
    /// Resize the block at `addr` to `new_size`, returning the (possibly
    /// new) guest address, or `None` if the request cannot be satisfied —
    /// `addr` is left untouched in that case.
    fn realloc(&self, addr: u64, new_size: u64) -> Option<u64>;
    /// Reserve a `length`-byte region aligned to `align`, returning its
    /// guest address, or `None` if the request cannot be satisfied.
    fn mmap(&self, length: u64, align: u64) -> Option<u64>;
    /// Release a `length`-byte region previously returned by `mmap` starting
    /// at `addr`. An unrecognized `addr` is simply ignored.
    fn munmap(&self, addr: u64, length: u64);
}

/// Everything an HLE function may touch: the emulated kernel (memory,
/// threads, filesystem, ...), the guest's address space, and the guest
/// allocator.
///
/// This is the dispatch-context milestone: before it existed, an HLE
/// function was a bare `fn(&[u64]) -> u64` with no way to read/write guest
/// pointers or reach a live [`xps5x_kernel::OrbisKernel`] — every stub was
/// necessarily a no-op that just logged and returned a plausible value. Now
/// every HLE call gets all three, so functions like `memcpy`/`strlen`/
/// `sceKernelMapFlexibleMemory` can do the real operation.
pub struct HleContext<'a> {
    /// The live emulated kernel (memory manager, thread manager, VFS, ...).
    pub kernel: &'a xps5x_kernel::OrbisKernel,
    /// The guest's address space, as seen from wherever this call
    /// originated (e.g. the runtime's mapped module image).
    pub mem: &'a dyn GuestMemory,
    /// The guest allocator backing `malloc`/`mmap` and friends. Not yet
    /// consumed by any HLE function body — see [`GuestAllocator`]'s doc
    /// comment.
    pub alloc: &'a dyn GuestAllocator,
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
        libsce_playgo::register(&registry);
        libsce_system_service::register(&registry);
        libsce_user_service::register(&registry);
        libsce_audio_out::register(&registry);
        libsce_save_data::register(&registry);
        libsce_common_dialog::register(&registry);
        libsce_app_content::register(&registry);
        libsce_np::register(&registry);

        info!(
            "HLE registry: {} functions registered",
            registry.functions.len()
        );
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
    pub fn call(
        &self,
        ctx: &HleContext,
        library: &str,
        function: &str,
        args: &[u64],
    ) -> Option<u64> {
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
        let Ok(addr) = usize::try_from(guest_addr) else {
            return false;
        };
        let buf = self.0.borrow();
        let Some(end) = addr.checked_add(out.len()) else {
            return false;
        };
        if end > buf.len() {
            return false;
        }
        out.copy_from_slice(&buf[addr..end]);
        true
    }

    fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else {
            return false;
        };
        let mut buf = self.0.borrow_mut();
        let Some(end) = addr.checked_add(data.len()) else {
            return false;
        };
        if end > buf.len() {
            return false;
        }
        buf[addr..end].copy_from_slice(data);
        true
    }
}

#[cfg(test)]
/// A minimal in-memory [`GuestAllocator`] test double, for unit tests that
/// need a complete [`HleContext`] but don't exercise allocation behavior
/// (nothing calls `ctx.alloc` yet — see [`GuestAllocator`]'s doc comment).
/// `alloc`/`mmap` are a bump allocator over a `Cell<u64>`; `free`/`munmap`
/// are no-ops; `realloc` always bumps a fresh block rather than reusing
/// `addr`.
pub(crate) struct TestAllocator(std::cell::Cell<u64>);

#[cfg(test)]
impl TestAllocator {
    pub(crate) fn new(base: u64) -> Self {
        Self(std::cell::Cell::new(base))
    }

    fn bump(&self, size: u64, align: u64) -> Option<u64> {
        let align = align.max(1);
        let cur = self.0.get();
        let aligned = cur.checked_add(align - 1)? & !(align - 1);
        let next = aligned.checked_add(size)?;
        self.0.set(next);
        Some(aligned)
    }
}

#[cfg(test)]
impl GuestAllocator for TestAllocator {
    fn alloc(&self, size: u64, align: u64) -> Option<u64> {
        self.bump(size, align)
    }

    fn free(&self, _addr: u64) {}

    fn realloc(&self, _addr: u64, new_size: u64) -> Option<u64> {
        self.bump(new_size, 1)
    }

    fn mmap(&self, length: u64, align: u64) -> Option<u64> {
        self.bump(length, align)
    }

    fn munmap(&self, _addr: u64, _length: u64) {}
}

/// Build an [`HleContext`] over a test kernel, [`TestMemory`], and
/// [`TestAllocator`]. Defined at the crate root (not inside `mod tests`) so
/// every submodule's own `#[cfg(test)] mod tests` can reach it as
/// `crate::test_ctx` — Rust visibility lets descendant modules see their
/// ancestors' private items.
#[cfg(test)]
pub(crate) fn test_ctx<'a>(
    kernel: &'a xps5x_kernel::OrbisKernel,
    mem: &'a TestMemory,
    alloc: &'a TestAllocator,
) -> HleContext<'a> {
    HleContext { kernel, mem, alloc }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_names_splits_library_and_function() {
        let registry = HleRegistry::new();
        let names = registry.registered_names();
        assert!(
            !names.is_empty(),
            "HleRegistry::new() should register some functions"
        );
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
        let alloc = TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
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
            assert!(
                result.is_some(),
                "{library}::{function} should return a value, not None"
            );
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
        assert_eq!(
            names,
            vec![("libFoo".to_string(), "someFunction".to_string())]
        );
    }
}
