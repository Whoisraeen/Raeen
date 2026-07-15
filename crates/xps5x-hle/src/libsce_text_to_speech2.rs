//! HLE lifecycle for the optional PS5 text-to-speech service.
//!
//! Speech synthesis is not fabricated: initialize/terminate maintain honest
//! per-process service state, while open/speak/status remain unresolved until
//! their guest ABI and audio delivery are implemented.

use std::sync::atomic::Ordering;

use crate::{HleContext, HleRegistry};

const SCE_OK: u64 = 0;

pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceTextToSpeech2",
        "sceTextToSpeech2Initialize",
        hle_initialize,
    );
    registry.register(
        "libSceTextToSpeech2",
        "sceTextToSpeech2Terminate",
        hle_terminate,
    );
}

fn hle_initialize(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.kernel
        .text_to_speech2_initialized
        .store(true, Ordering::Release);
    SCE_OK
}

fn hle_terminate(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.kernel
        .text_to_speech2_initialized
        .store(false, Ordering::Release);
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    #[test]
    fn lifecycle_state_is_per_kernel_process() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let other_kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_initialize(&ctx, &[]), SCE_OK);
        assert!(kernel.text_to_speech2_initialized.load(Ordering::Acquire));
        assert!(
            !other_kernel
                .text_to_speech2_initialized
                .load(Ordering::Acquire)
        );

        assert_eq!(hle_terminate(&ctx, &[]), SCE_OK);
        assert!(!kernel.text_to_speech2_initialized.load(Ordering::Acquire));
    }
}
