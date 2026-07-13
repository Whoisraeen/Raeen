//! GNM API translation — Sony's low-level graphics API.
//!
//! GNM is Sony's proprietary graphics API for PS5. Games submit GPU
//! work by building PM4 (Packet Manager 4) command buffers that contain
//! register writes, draw commands, compute dispatches, and sync operations.
//!
//! This module decodes PM4 packets and translates them to Vulkan.

pub mod command_buffer;
pub mod draw;
pub mod compute;
pub mod registers;

use tracing::{debug, info};

/// GNM context — holds the current GPU state for translation.
pub struct GnmContext {
    /// Current GPU register state.
    pub registers: registers::RegisterState,
    /// Statistics.
    pub stats: GnmStats,
}

/// GNM translation statistics.
#[derive(Debug, Default)]
pub struct GnmStats {
    pub pm4_packets_decoded: u64,
    pub draw_calls: u64,
    pub compute_dispatches: u64,
    pub register_writes: u64,
    pub unknown_opcodes: u64,
}

impl GnmContext {
    pub fn new() -> Self {
        info!("Initializing GNM context");
        Self {
            registers: registers::RegisterState::new(),
            stats: GnmStats::default(),
        }
    }

    /// Process a PM4 command buffer.
    ///
    /// Reads PM4 packets from the buffer and translates each one.
    pub fn process_command_buffer(&mut self, data: &[u32]) {
        let mut offset = 0;

        while offset < data.len() {
            let header = data[offset];
            let packet_type = (header >> 30) & 0x3;

            match packet_type {
                0 => {
                    // Type 0: Register write.
                    let reg = header & 0xFFFF;
                    let count = ((header >> 16) & 0x3FFF) + 1;

                    for i in 0..count as usize {
                        if offset + 1 + i < data.len() {
                            self.registers.write(reg + i as u32, data[offset + 1 + i]);
                            self.stats.register_writes += 1;
                        }
                    }

                    offset += 1 + count as usize;
                    self.stats.pm4_packets_decoded += 1;
                }
                2 => {
                    // Type 2: NOP (filler/padding).
                    offset += 1;
                    self.stats.pm4_packets_decoded += 1;
                }
                3 => {
                    // Type 3: GPU command (draw, dispatch, sync, etc.).
                    let opcode = (header >> 8) & 0xFF;
                    let count = ((header >> 16) & 0x3FFF) + 1;
                    let body_start = offset + 1;
                    let body_end = (body_start + count as usize).min(data.len());
                    let body = &data[body_start..body_end];

                    self.handle_type3_packet(opcode, body);

                    offset = body_end;
                    self.stats.pm4_packets_decoded += 1;
                }
                _ => {
                    debug!("Unknown PM4 packet type {} at offset {}", packet_type, offset);
                    offset += 1;
                    self.stats.unknown_opcodes += 1;
                }
            }
        }
    }

    /// Handle a Type 3 PM4 packet (GPU command).
    fn handle_type3_packet(&mut self, opcode: u32, body: &[u32]) {
        match opcode {
            // ─── Draw commands ─────────────────────────────
            command_buffer::PM4_DRAW_INDEX_2 => {
                debug!("PM4: DRAW_INDEX_2");
                self.stats.draw_calls += 1;
            }
            command_buffer::PM4_DRAW_INDEX_AUTO => {
                debug!("PM4: DRAW_INDEX_AUTO");
                self.stats.draw_calls += 1;
            }
            command_buffer::PM4_DRAW_INDEX_OFFSET_2 => {
                debug!("PM4: DRAW_INDEX_OFFSET_2");
                self.stats.draw_calls += 1;
            }

            // ─── Compute ───────────────────────────────────
            command_buffer::PM4_DISPATCH_DIRECT => {
                debug!("PM4: DISPATCH_DIRECT");
                self.stats.compute_dispatches += 1;
            }

            // ─── Register writes ───────────────────────────
            command_buffer::PM4_SET_CONTEXT_REG => {
                if body.len() >= 2 {
                    let reg_offset = body[0] & 0xFFFF;
                    for (i, &value) in body[1..].iter().enumerate() {
                        self.registers.write_context(reg_offset + i as u32, value);
                        self.stats.register_writes += 1;
                    }
                }
            }
            command_buffer::PM4_SET_SH_REG => {
                if body.len() >= 2 {
                    let reg_offset = body[0] & 0xFFFF;
                    for (i, &value) in body[1..].iter().enumerate() {
                        self.registers.write_sh(reg_offset + i as u32, value);
                        self.stats.register_writes += 1;
                    }
                }
            }
            command_buffer::PM4_SET_UCONFIG_REG => {
                if body.len() >= 2 {
                    let reg_offset = body[0] & 0xFFFF;
                    for (i, &value) in body[1..].iter().enumerate() {
                        self.registers.write_uconfig(reg_offset + i as u32, value);
                        self.stats.register_writes += 1;
                    }
                }
            }

            // ─── Synchronization ───────────────────────────
            command_buffer::PM4_EVENT_WRITE_EOP => {
                debug!("PM4: EVENT_WRITE_EOP (End of Pipe)");
            }
            command_buffer::PM4_EVENT_WRITE_EOS => {
                debug!("PM4: EVENT_WRITE_EOS (End of Shader)");
            }
            command_buffer::PM4_WAIT_REG_MEM => {
                debug!("PM4: WAIT_REG_MEM");
            }
            command_buffer::PM4_ACQUIRE_MEM => {
                debug!("PM4: ACQUIRE_MEM (cache flush/invalidate)");
            }
            command_buffer::PM4_RELEASE_MEM => {
                debug!("PM4: RELEASE_MEM");
            }

            // ─── Indirect buffer ───────────────────────────
            command_buffer::PM4_INDIRECT_BUFFER => {
                debug!("PM4: INDIRECT_BUFFER (chained command buffer)");
                // In a full implementation, follow the IB pointer and
                // recursively process the indirect command buffer.
            }

            // ─── DMA ───────────────────────────────────────
            command_buffer::PM4_DMA_DATA => {
                debug!("PM4: DMA_DATA (memory copy)");
            }

            // ─── Unknown ──────────────────────────────────
            _ => {
                debug!("PM4: Unknown opcode {:#x} ({} dwords)", opcode, body.len());
                self.stats.unknown_opcodes += 1;
            }
        }
    }
}

impl Default for GnmContext {
    fn default() -> Self {
        Self::new()
    }
}
