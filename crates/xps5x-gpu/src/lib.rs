//! # XPS5X GPU
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

pub mod backend;
pub mod gnm;
pub mod metal;
pub mod shader;
pub mod texture;
pub mod vulkan;

pub use backend::{BackendKind, GpuBackend, ShaderFormat, create_backend, default_backend_kind};
