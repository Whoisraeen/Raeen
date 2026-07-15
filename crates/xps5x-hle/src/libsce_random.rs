//! HLE `libSceRandom` — host-backed cryptographic random bytes.

use crate::{HleContext, HleRegistry};

pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceRandom",
        "sceRandomGetRandomNumber",
        hle_get_random_number,
    );
}

/// `sceRandomGetRandomNumber(void *output, size_t length)` fills the complete
/// guest range from the host OS cryptographic RNG. Work is chunked so a guest
/// cannot force a proportional host allocation with a very large request.
fn hle_get_random_number(ctx: &HleContext, args: &[u64]) -> u64 {
    const SCE_OK: u64 = 0;
    const SCE_KERNEL_ERROR_EIO: u64 = 0x8002_0005;
    const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E;
    const CHUNK_SIZE: usize = 4096;

    let output = args.first().copied().unwrap_or(0);
    let length = args.get(1).copied().unwrap_or(0);
    if length == 0 {
        return SCE_OK;
    }
    if output == 0 || output.checked_add(length).is_none() {
        return SCE_KERNEL_ERROR_EFAULT;
    }

    let mut chunk = [0u8; CHUNK_SIZE];
    let mut written = 0u64;
    while written < length {
        let count = (length - written).min(CHUNK_SIZE as u64) as usize;
        if getrandom::fill(&mut chunk[..count]).is_err() {
            return SCE_KERNEL_ERROR_EIO;
        }
        if !ctx.mem.write(output + written, &chunk[..count]) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        written += count as u64;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn random_number_fills_guest_memory_and_validates_the_range() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_random_number(&ctx, &[0x20, 32]), 0);
        let mut first = [0u8; 32];
        assert!(mem.read(0x20, &mut first));
        assert_ne!(first, [0; 32]);

        assert_eq!(hle_get_random_number(&ctx, &[0x20, 32]), 0);
        let mut second = [0u8; 32];
        assert!(mem.read(0x20, &mut second));
        assert_ne!(first, second);

        assert_ne!(hle_get_random_number(&ctx, &[0, 32]), 0);
        assert_ne!(hle_get_random_number(&ctx, &[0xF8, 32]), 0);
    }
}
