//! PM4 command buffer definitions and opcode constants.
//!
//! AMD's PM4 (Packet Manager 4) protocol defines how the CPU
//! communicates with the GPU's Command Processor. These opcodes
//! are shared between GCN and RDNA architectures.

// ─── Draw commands ─────────────────────────────────────────
/// Draw with index buffer, count in packet body.
pub const PM4_DRAW_INDEX_2: u32 = 0x27;
/// Draw with auto-generated indices.
pub const PM4_DRAW_INDEX_AUTO: u32 = 0x2D;
/// Draw with index buffer and offset.
pub const PM4_DRAW_INDEX_OFFSET_2: u32 = 0x30;
/// Draw indirect (GPU-driven).
pub const PM4_DRAW_INDEX_INDIRECT: u32 = 0x38;
/// Draw indirect multi (GPU-driven, multiple draws).
pub const PM4_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
/// Draw with per-instance data.
pub const PM4_DRAW_INDEX_MULTI_INST: u32 = 0x2A;

// ─── Compute ───────────────────────────────────────────────
/// Direct compute dispatch (threadgroup counts in packet).
pub const PM4_DISPATCH_DIRECT: u32 = 0x15;
/// Indirect compute dispatch.
pub const PM4_DISPATCH_INDIRECT: u32 = 0x18;

// ─── Register writes ───────────────────────────────────────
/// Write to context registers (rendering state).
pub const PM4_SET_CONTEXT_REG: u32 = 0x69;
/// Write to SH (shader) registers.
pub const PM4_SET_SH_REG: u32 = 0x76;
/// Write to UCONFIG registers.
pub const PM4_SET_UCONFIG_REG: u32 = 0x79;

// ─── Synchronization ──────────────────────────────────────
/// Event write: End of Pipe (fence signal).
pub const PM4_EVENT_WRITE_EOP: u32 = 0x47;
/// Event write: End of Shader.
pub const PM4_EVENT_WRITE_EOS: u32 = 0x48;
/// Wait for register/memory value.
pub const PM4_WAIT_REG_MEM: u32 = 0x3C;
/// Acquire memory (cache flush/invalidate).
pub const PM4_ACQUIRE_MEM: u32 = 0x58;
/// Release memory (make writes visible).
pub const PM4_RELEASE_MEM: u32 = 0x49;
/// Write data to memory/register.
pub const PM4_WRITE_DATA: u32 = 0x37;

// ─── Indirect execution ───────────────────────────────────
/// Execute commands from an indirect buffer.
pub const PM4_INDIRECT_BUFFER: u32 = 0x3F;
/// Conditional execution.
pub const PM4_COND_EXEC: u32 = 0x22;

// ─── DMA ───────────────────────────────────────────────────
/// DMA data transfer between GPU memory regions.
pub const PM4_DMA_DATA: u32 = 0x50;
/// Copy data.
pub const PM4_COPY_DATA: u32 = 0x40;

// ─── Misc ──────────────────────────────────────────────────
/// No operation.
pub const PM4_NOP: u32 = 0x10;
/// Prefetch L2 cache.
pub const PM4_PFP_SYNC_ME: u32 = 0x31;
/// Surface sync.
pub const PM4_SURFACE_SYNC: u32 = 0x43;
/// Index type and primitive topology.
pub const PM4_INDEX_TYPE: u32 = 0x2A;
/// Set predication.
pub const PM4_SET_PREDICATION: u32 = 0x20;
/// Context control (GPU context save/restore).
pub const PM4_CONTEXT_CONTROL: u32 = 0x28;

/// Decode a PM4 packet header.
#[derive(Debug, Clone, Copy)]
pub struct Pm4Header {
    /// Packet type (0, 2, or 3).
    pub packet_type: u8,
    /// Opcode (Type 3 only).
    pub opcode: u32,
    /// Number of DWORDs in the packet body.
    pub count: u32,
}

impl Pm4Header {
    /// Parse a PM4 header from a raw u32.
    pub fn from_raw(raw: u32) -> Self {
        let packet_type = ((raw >> 30) & 0x3) as u8;
        let opcode = if packet_type == 3 {
            (raw >> 8) & 0xFF
        } else {
            0
        };
        let count = if packet_type == 3 || packet_type == 0 {
            ((raw >> 16) & 0x3FFF) + 1
        } else {
            0
        };

        Self {
            packet_type,
            opcode,
            count,
        }
    }

    /// Get the opcode name for debugging.
    pub fn opcode_name(&self) -> &'static str {
        match self.opcode {
            PM4_DRAW_INDEX_2 => "DRAW_INDEX_2",
            PM4_DRAW_INDEX_AUTO => "DRAW_INDEX_AUTO",
            PM4_DISPATCH_DIRECT => "DISPATCH_DIRECT",
            PM4_SET_CONTEXT_REG => "SET_CONTEXT_REG",
            PM4_SET_SH_REG => "SET_SH_REG",
            PM4_SET_UCONFIG_REG => "SET_UCONFIG_REG",
            PM4_EVENT_WRITE_EOP => "EVENT_WRITE_EOP",
            PM4_WAIT_REG_MEM => "WAIT_REG_MEM",
            PM4_ACQUIRE_MEM => "ACQUIRE_MEM",
            PM4_RELEASE_MEM => "RELEASE_MEM",
            PM4_INDIRECT_BUFFER => "INDIRECT_BUFFER",
            PM4_DMA_DATA => "DMA_DATA",
            PM4_NOP => "NOP",
            _ => "UNKNOWN",
        }
    }
}
