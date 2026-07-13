//! GPU register state machine.
//!
//! Tracks the state of hundreds of GPU registers that control
//! rendering pipeline configuration: blend, rasterization, depth,
//! stencil, viewport, scissor, etc.
//!
//! Register writes from PM4 packets update this state, which is
//! then read when translating draw calls to Vulkan.

use std::collections::HashMap;
use tracing::debug;

/// Register space bases (RDNA2).
const CONTEXT_REG_BASE: u32 = 0xA000;
const SH_REG_BASE: u32 = 0x2C00;
const UCONFIG_REG_BASE: u32 = 0xC000;

// ─── Well-known context registers ──────────────────────────
/// Depth buffer control.
pub const DB_DEPTH_CONTROL: u32 = 0xA200;
/// Stencil control.
pub const DB_STENCIL_CONTROL: u32 = 0xA10C;
/// Color buffer control.
pub const CB_COLOR_CONTROL: u32 = 0xA202;
/// Blend control (per render target, 0-7).
pub const CB_BLEND0_CONTROL: u32 = 0xA1E0;
/// Polygon rasterization mode.
pub const PA_SU_SC_MODE_CNTL: u32 = 0xA205;
/// Clip control.
pub const PA_CL_CLIP_CNTL: u32 = 0xA204;
/// Viewport transform (X scale).
pub const PA_CL_VPORT_XSCALE_0: u32 = 0xA10F;
/// Scissor rect TL.
pub const PA_SC_SCREEN_SCISSOR_TL: u32 = 0xA00C;
/// Scissor rect BR.
pub const PA_SC_SCREEN_SCISSOR_BR: u32 = 0xA00D;

/// Tracks the full GPU register state.
pub struct RegisterState {
    /// Context registers (rendering state).
    context_regs: HashMap<u32, u32>,
    /// SH registers (shader state).
    sh_regs: HashMap<u32, u32>,
    /// UCONFIG registers (GPU-global config).
    uconfig_regs: HashMap<u32, u32>,
}

impl RegisterState {
    pub fn new() -> Self {
        Self {
            context_regs: HashMap::with_capacity(1024),
            sh_regs: HashMap::with_capacity(512),
            uconfig_regs: HashMap::with_capacity(256),
        }
    }

    /// Write a raw register value.
    pub fn write(&mut self, addr: u32, value: u32) {
        if (CONTEXT_REG_BASE..CONTEXT_REG_BASE + 0x1000).contains(&addr) {
            self.context_regs.insert(addr, value);
        } else if (SH_REG_BASE..SH_REG_BASE + 0x400).contains(&addr) {
            self.sh_regs.insert(addr, value);
        } else if addr >= UCONFIG_REG_BASE {
            self.uconfig_regs.insert(addr, value);
        }
    }

    /// Write a context register (offset from CONTEXT_REG_BASE).
    pub fn write_context(&mut self, offset: u32, value: u32) {
        let addr = CONTEXT_REG_BASE + offset;
        self.context_regs.insert(addr, value);
    }

    /// Write a shader register (offset from SH_REG_BASE).
    pub fn write_sh(&mut self, offset: u32, value: u32) {
        let addr = SH_REG_BASE + offset;
        self.sh_regs.insert(addr, value);
    }

    /// Write a UCONFIG register (offset from UCONFIG_REG_BASE).
    pub fn write_uconfig(&mut self, offset: u32, value: u32) {
        let addr = UCONFIG_REG_BASE + offset;
        self.uconfig_regs.insert(addr, value);
    }

    /// Read a context register.
    pub fn read_context(&self, addr: u32) -> u32 {
        self.context_regs.get(&addr).copied().unwrap_or(0)
    }

    /// Read a shader register.
    pub fn read_sh(&self, addr: u32) -> u32 {
        self.sh_regs.get(&addr).copied().unwrap_or(0)
    }

    /// Check if depth testing is enabled.
    pub fn is_depth_test_enabled(&self) -> bool {
        let db_control = self.read_context(DB_DEPTH_CONTROL);
        db_control & 0x1 != 0
    }

    /// Check if depth writing is enabled.
    pub fn is_depth_write_enabled(&self) -> bool {
        let db_control = self.read_context(DB_DEPTH_CONTROL);
        db_control & 0x2 != 0
    }

    /// Get the polygon mode (fill, line, point).
    pub fn polygon_mode(&self) -> PolygonMode {
        let mode = self.read_context(PA_SU_SC_MODE_CNTL);
        match (mode >> 3) & 0x3 {
            0 => PolygonMode::Fill,
            1 => PolygonMode::Line,
            2 => PolygonMode::Point,
            _ => PolygonMode::Fill,
        }
    }

    /// Get the cull mode.
    pub fn cull_mode(&self) -> CullMode {
        let mode = self.read_context(PA_SU_SC_MODE_CNTL);
        let cull_front = mode & 0x1 != 0;
        let cull_back = mode & 0x2 != 0;
        match (cull_front, cull_back) {
            (false, false) => CullMode::None,
            (true, false) => CullMode::Front,
            (false, true) => CullMode::Back,
            (true, true) => CullMode::FrontAndBack,
        }
    }

    /// Reset all registers to defaults.
    pub fn reset(&mut self) {
        self.context_regs.clear();
        self.sh_regs.clear();
        self.uconfig_regs.clear();
        debug!("GPU registers reset to defaults");
    }
}

/// Polygon fill mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

/// Face culling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
    FrontAndBack,
}

impl Default for RegisterState {
    fn default() -> Self {
        Self::new()
    }
}
