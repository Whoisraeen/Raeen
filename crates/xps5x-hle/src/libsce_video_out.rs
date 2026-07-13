//! HLE libSceVideoOut — Display output management.
//!
//! Manages video output handles and buffer flipping (present).

use crate::HleRegistry;
use tracing::debug;

/// Register libSceVideoOut HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceVideoOut", "sceVideoOutOpen", hle_video_out_open);
    registry.register("libSceVideoOut", "sceVideoOutClose", hle_video_out_close);
    registry.register("libSceVideoOut", "sceVideoOutSetFlipRate", hle_video_out_set_flip_rate);
    registry.register("libSceVideoOut", "sceVideoOutRegisterBuffers", hle_video_out_register_buffers);
    registry.register("libSceVideoOut", "sceVideoOutSubmitFlip", hle_video_out_submit_flip);
}

fn hle_video_out_open(args: &[u64]) -> u64 {
    debug!("sceVideoOutOpen(userId={}, busType={}, index={})", args[0], args[1], args[2]);
    1 // Return handle = 1.
}

fn hle_video_out_close(args: &[u64]) -> u64 {
    debug!("sceVideoOutClose(handle={})", args[0]);
    0
}

fn hle_video_out_set_flip_rate(args: &[u64]) -> u64 {
    debug!("sceVideoOutSetFlipRate(handle={}, rate={})", args[0], args[1]);
    0
}

fn hle_video_out_register_buffers(args: &[u64]) -> u64 {
    debug!("sceVideoOutRegisterBuffers(handle={}, ...)", args[0]);
    0
}

fn hle_video_out_submit_flip(args: &[u64]) -> u64 {
    debug!("sceVideoOutSubmitFlip(handle={}, bufferIndex={})", args[0], args[1]);
    0
}
