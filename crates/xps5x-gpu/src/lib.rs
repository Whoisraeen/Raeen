//! # XPS5X GPU
//!
//! GPU command translation layer for the PS5 emulator.
//!
//! Translates Sony's proprietary GNM graphics API commands (submitted
//! as AMD PM4 packets) into Vulkan API calls. Also includes a shader
//! recompiler that converts PS5's precompiled RDNA2 ISA shader binaries
//! to SPIR-V bytecode for the host GPU.
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

pub mod gnm;
pub mod shader;
pub mod vulkan;
pub mod texture;
