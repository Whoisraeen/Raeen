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

/// Builds a PM4 command buffer — the inverse of [`Pm4Header::from_raw`] and the
/// decoder in `GnmContext::process_command_buffer`. This is the encode side the
/// GPU-command layer (e.g. a future `libSceAgc` port) and any in-emulator
/// command-buffer construction build on; it is exercised end-to-end by feeding
/// its output back through the decoder (see the `gnm` round-trip test).
#[derive(Default)]
pub struct Pm4Writer {
    words: Vec<u32>,
}

impl Pm4Writer {
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// The encoded command-buffer words.
    pub fn as_slice(&self) -> &[u32] {
        &self.words
    }

    pub fn into_words(self) -> Vec<u32> {
        self.words
    }

    /// A Type 3 header: `type(3) | (count-1) | opcode`. `count` is the number
    /// of body DWORDs that follow (must be ≥ 1).
    fn type3_header(opcode: u32, count: u32) -> u32 {
        debug_assert!(count >= 1);
        (3 << 30) | (((count - 1) & 0x3FFF) << 16) | ((opcode & 0xFF) << 8)
    }

    /// A Type 0 header: `type(0) | (count-1) | reg`. Writes `count` DWORDs to
    /// consecutive registers starting at `reg`.
    fn type0_header(reg: u32, count: u32) -> u32 {
        debug_assert!(count >= 1);
        (((count - 1) & 0x3FFF) << 16) | (reg & 0xFFFF)
    }

    /// Emit a raw Type 3 packet with an explicit body.
    pub fn type3(&mut self, opcode: u32, body: &[u32]) {
        self.words
            .push(Self::type3_header(opcode, body.len().max(1) as u32));
        self.words.extend_from_slice(body);
    }

    /// `SET_CONTEXT_REG`: write `values` to context registers starting at the
    /// base-relative `reg_offset` (the decoder adds `CONTEXT_REG_BASE`).
    pub fn set_context_reg(&mut self, reg_offset: u32, values: &[u32]) {
        let mut body = Vec::with_capacity(1 + values.len());
        body.push(reg_offset);
        body.extend_from_slice(values);
        self.type3(PM4_SET_CONTEXT_REG, &body);
    }

    /// `SET_SH_REG`: write `values` to SH registers from base-relative `reg_offset`.
    pub fn set_sh_reg(&mut self, reg_offset: u32, values: &[u32]) {
        let mut body = Vec::with_capacity(1 + values.len());
        body.push(reg_offset);
        body.extend_from_slice(values);
        self.type3(PM4_SET_SH_REG, &body);
    }

    /// `DRAW_INDEX_AUTO`: a non-indexed draw of `index_count` vertices.
    pub fn draw_index_auto(&mut self, index_count: u32) {
        self.type3(PM4_DRAW_INDEX_AUTO, &[index_count, 0]);
    }

    /// `DISPATCH_DIRECT`: a compute dispatch of `(x, y, z)` threadgroups.
    pub fn dispatch_direct(&mut self, x: u32, y: u32, z: u32) {
        self.type3(PM4_DISPATCH_DIRECT, &[x, y, z, 0]);
    }

    /// A Type 0 register write to `reg`..`reg+values.len()`.
    pub fn write_type0(&mut self, reg: u32, values: &[u32]) {
        self.words
            .push(Self::type0_header(reg, values.len().max(1) as u32));
        self.words.extend_from_slice(values);
    }

    /// A Type 2 NOP (single-word filler).
    pub fn nop(&mut self) {
        self.words.push(2 << 30);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_headers_decode_to_the_same_fields() {
        // A Type 3 SET_CONTEXT_REG with a 3-DWORD body (reg_offset + 2 values).
        let mut w = Pm4Writer::new();
        w.set_context_reg(0x10, &[0xAAAA, 0xBBBB]);
        let words = w.as_slice();
        let h = Pm4Header::from_raw(words[0]);
        assert_eq!(h.packet_type, 3);
        assert_eq!(h.opcode, PM4_SET_CONTEXT_REG);
        assert_eq!(h.count, 3, "body = reg_offset + 2 values");
        assert_eq!(&words[1..], &[0x10, 0xAAAA, 0xBBBB]);

        // A Type 0 write of 2 registers round-trips through the header parser.
        let mut w0 = Pm4Writer::new();
        w0.write_type0(0xA000, &[1, 2]);
        let h0 = Pm4Header::from_raw(w0.as_slice()[0]);
        assert_eq!(h0.packet_type, 0);
        assert_eq!(h0.count, 2);
        assert_eq!(
            w0.as_slice()[0] & 0xFFFF,
            0xA000,
            "register base in low bits"
        );
    }

    #[test]
    fn draw_and_dispatch_encode_the_right_opcodes() {
        let mut w = Pm4Writer::new();
        w.draw_index_auto(3);
        w.dispatch_direct(4, 1, 1);
        w.nop();
        let words = w.as_slice();
        assert_eq!(Pm4Header::from_raw(words[0]).opcode, PM4_DRAW_INDEX_AUTO);
        // draw body = [count, initiator] → next packet starts at index 3.
        assert_eq!(Pm4Header::from_raw(words[3]).opcode, PM4_DISPATCH_DIRECT);
        // dispatch body = 4 dwords → NOP (type 2) at index 3+1+4 = 8.
        assert_eq!(Pm4Header::from_raw(words[8]).packet_type, 2);
    }
}
