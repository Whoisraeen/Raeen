//! Faithful Rust port of Kyty `emulator/src/Graphics` (MIT © 2021 InoriRus).
//!
//! Phase 5 of the Kyty port plan. The shader recompiler landed first:
//! `shader_parse` (GCN instruction decode), `shader` (binary info / usages),
//! `shader_spirv` (SPIR-V generation).
//!
//! The PM4 command processor follows: [`pm4`] (packet codec + opcode /
//! register indices from `Pm4.h`), [`hw_regs`] (the `HardwareContext.h`
//! register model), and [`run`] (`GraphicsRun.cpp`'s `CommandProcessor`).
//!
//! This crate deliberately carries **no Vulkan dependency** — Kyty's
//! `GraphicsRender` layer is not ported here. [`run::CommandProcessor`]
//! terminates at the [`run::DrawSink`] trait, which `raeen-gpu` implements.

pub mod hw_regs;
pub mod pm4;
pub mod run;
pub mod shader;
pub mod spirv_asm;
