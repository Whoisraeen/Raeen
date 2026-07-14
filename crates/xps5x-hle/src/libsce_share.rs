//! HLE libSceShareUtility — the Share (broadcast/screenshot) content-param
//! handshake.
//!
//! A faithful Rust port of SharpEmu's `ShareExports` (GPL-2.0). Note the
//! SharpEmu library name is `libSceShareUtility` (not `libSceShare`). A title
//! initializes the Share system and sets a "content param" (a UTF-8 string the
//! Share overlay would display). XPS5X has no Share/broadcast backend, so this
//! is an honest handshake stub: `Initialize` validates + records an
//! initialized flag, and `SetContentParam` reads and retains the string. No
//! broadcast or screenshot is ever produced.
//!
//! The generic `OrbisGen2Result` codes map to the real Orbis `EINVAL`/`EFAULT`
//! (`0x8002_0016`/`0x8002_000E`) as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

/// Maximum bytes scanned for the (NUL-terminated) content param string.
const MAX_CONTENT_PARAM_BYTES: u64 = 4096;

// SharpEmu's `_initialized` flag and `_contentParam` retained string.
static INITIALIZED: AtomicI32 = AtomicI32::new(0);
static CONTENT_PARAM: Mutex<String> = Mutex::new(String::new());

/// Register the libSceShareUtility functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceShareUtility", "sceShareInitialize", hle_initialize);
    registry.register(
        "libSceShareUtility",
        "sceShareSetContentParam",
        hle_set_content_param,
    );
}

/// `sceShareInitialize(memorySize, priority, affinityMask)`: a zero memory size
/// is an invalid-argument error; otherwise the library is marked initialized.
fn hle_initialize(_ctx: &HleContext, args: &[u64]) -> u64 {
    let memory_size = args.first().copied().unwrap_or(0);
    if memory_size == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    INITIALIZED.store(1, Ordering::Relaxed);
    OK
}

/// `sceShareSetContentParam(contentParam *)`: reads the NUL-terminated UTF-8
/// content-param string (up to 4096 bytes) and retains it. A null pointer is an
/// invalid-argument error; an unreadable / unterminated string is a memory
/// fault (matching SharpEmu, which fails if no NUL is found within the bound).
fn hle_set_content_param(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    if addr == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut bytes = Vec::new();
    let mut one = [0u8; 1];
    let mut found_nul = false;
    for index in 0..MAX_CONTENT_PARAM_BYTES {
        if !ctx.mem.read(addr + index, &mut one) {
            return SCE_ERROR_MEMORY_FAULT;
        }
        if one[0] == 0 {
            found_nul = true;
            break;
        }
        bytes.push(one[0]);
    }
    if !found_nul {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if let Ok(mut param) = CONTENT_PARAM.lock() {
        *param = String::from_utf8_lossy(&bytes).into_owned();
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn initialize_validates_memory_size() {
        INITIALIZED.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_initialize(&ctx, &[0, 0, 0]), SCE_ERROR_INVALID_ARGUMENT);
        assert_eq!(hle_initialize(&ctx, &[0x1000, 0, 0]), OK);
        assert_eq!(INITIALIZED.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn set_content_param_reads_and_retains_string() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_set_content_param(&ctx, &[0]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        // Write a NUL-terminated string into guest memory.
        assert!(mem.write(0x20, b"hello\0"));
        assert_eq!(hle_set_content_param(&ctx, &[0x20]), OK);
        assert_eq!(CONTENT_PARAM.lock().unwrap().as_str(), "hello");
    }

    #[test]
    fn set_content_param_faults_on_unreadable() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // An address whose whole 4096-byte scan window is out of bounds faults.
        assert_eq!(
            hle_set_content_param(&ctx, &[0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
    }
}
