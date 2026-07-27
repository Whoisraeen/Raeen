//! # Raeen GPU
//!
//! GPU command translation layer for the PS5 emulator.
//!
//! Translates Sony's proprietary GNM graphics API commands (submitted
//! as AMD PM4 packets) into host graphics API calls through a pluggable
//! [`GpuBackend`](backend::GpuBackend) — Vulkan on Windows/Linux and
//! Metal on macOS. Also includes a shader recompiler that converts PS5's
//! precompiled RDNA2 ISA shader binaries into the backend's shader format
//! ([`ShaderFormat`](backend::ShaderFormat): SPIR-V for Vulkan, MSL for Metal).
//!
//! ## Pipeline
//!
//! ```text
//! PS5 Game → GNM API → PM4 Command Buffer
//!                          ↓
//!                    PM4 Packet Decoder
//!                          ↓
//!                    Register State Machine
//!                          ↓
//!                    Vulkan Command Translation
//!                          ↓
//!                    Host GPU (Vulkan 1.3)
//! ```

pub mod agc;
pub mod agc_exec;
pub mod backend;
pub mod contracts;
mod diagnostics;
pub(crate) mod draw_translate;
pub mod frame_ipc;
pub mod gnm;
mod guest_mem;
pub mod metal;
pub mod present_plugin;
pub mod shader;
pub mod shader_bridge;
pub(crate) mod shader_fetch;
pub mod spirv_gate;
pub mod texture;
pub mod vulkan;

#[allow(deprecated)]
pub use agc_exec::build_m2_draw_dcb;
pub use agc_exec::{
    AgcGpuSession, GpuProcessSession, PresentTiming, ScissorHalf, build_cp_draw_dcb,
};
pub use backend::{BackendKind, GpuBackend, ShaderFormat, create_backend, default_backend_kind};
pub use contracts::{ShaderMappedData, ShaderSemantic, ShaderSharp, ShaderUserData};
pub use guest_mem::GpuGuestMemory;
pub use present_plugin::{
    Capabilities, PluginFrame, PluginInfo, PluginOutput, PresentContext, PresentFrame,
    PresentPlugin,
};
pub use shader_fetch::ShaderCacheStats;
pub use vulkan::offscreen::RenderedImage;
