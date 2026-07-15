//! PS5 AGC (Gen5) PM4 command-stream decoding.
//!
//! AGC uses a type-3 header whose encoded length is the **total** packet size
//! minus two. This differs from the older GNM decoder's body-count convention,
//! so retail PS5 DCBs must pass through this decoder before Vulkan execution.

use thiserror::Error;

const IT_NOP: u32 = 0x10;
const IT_DRAW_INDEX_2: u32 = 0x27;
const IT_DRAW_INDEX_AUTO: u32 = 0x2d;
const IT_DRAW_INDEX_OFFSET_2: u32 = 0x35;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_DRAW_INDEX_INDIRECT: u32 = 0x25;
const R_DRAW_INDEX_AUTO: u32 = 0x04;
const R_DRAW_INDEX: u32 = 0x03;
const R_FLIP: u32 = 0x17;
const R_DISPATCH_DIRECT: u32 = 0x08;
const R_WRITE_DATA: u32 = 0x15;
const R_RELEASE_MEM: u32 = 0x18;
const R_WAIT_MEM_32: u32 = 0x0a;
const IT_DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_DISPATCH_DRAW: u32 = 0x8d;
const IT_EVENT_WRITE: u32 = 0x46;

/// One decoded Gen5 PM4 packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcPacket {
    /// DWORD offset from the start of the submitted DCB.
    pub offset: u32,
    /// Total packet length in DWORDs.
    pub dwords: u32,
    /// Type-3 opcode, or zero for a type-2 filler.
    pub opcode: u32,
    /// AGC's six-bit NOP sub-discriminator.
    pub register: u32,
}

/// A flip embedded in a submitted DCB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcFlip {
    pub video_out_handle: u32,
    pub display_buffer_index: u32,
    pub flip_mode: u32,
    pub flip_arg: u64,
}

/// One guest-memory write requested by a synchronization/data packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgcMemoryWrite {
    /// DWORD offset of the packet that produced this write.
    pub packet_offset: u32,
    pub address: u64,
    pub data: Vec<u8>,
}

/// One Gen5 32-bit memory comparison that gates later command execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcWait32 {
    pub packet_offset: u32,
    pub address: u64,
    pub mask: u32,
    pub function: u32,
    pub reference: u32,
}

/// Structural facts extracted from a complete DCB submission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgcSubmission {
    pub packets: Vec<AgcPacket>,
    pub draw_packets: u32,
    pub dispatch_packets: u32,
    pub flips: Vec<AgcFlip>,
    pub memory_writes: Vec<AgcMemoryWrite>,
    pub waits32: Vec<AgcWait32>,
    /// Event ids signaled by standard `EVENT_WRITE` packets.
    pub events: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AgcDecodeError {
    #[error("unsupported PM4 packet type {packet_type} at DWORD {offset}")]
    UnsupportedPacketType { offset: u32, packet_type: u32 },
    #[error("truncated PM4 packet at DWORD {offset}: needs {needed}, has {remaining}")]
    Truncated {
        offset: u32,
        needed: u32,
        remaining: u32,
    },
}

/// Decode one complete AGC draw command buffer.
pub fn decode_submission(words: &[u32]) -> Result<AgcSubmission, AgcDecodeError> {
    let mut result = AgcSubmission::default();
    let mut offset = 0usize;
    while offset < words.len() {
        let header = words[offset];
        let packet_type = header >> 30;
        if packet_type == 2 {
            result.packets.push(AgcPacket {
                offset: offset as u32,
                dwords: 1,
                opcode: 0,
                register: 0,
            });
            offset += 1;
            continue;
        }
        if packet_type != 3 {
            return Err(AgcDecodeError::UnsupportedPacketType {
                offset: offset as u32,
                packet_type,
            });
        }

        let dwords = ((header >> 16) & 0x3fff) + 2;
        let remaining = (words.len() - offset) as u32;
        if dwords > remaining {
            return Err(AgcDecodeError::Truncated {
                offset: offset as u32,
                needed: dwords,
                remaining,
            });
        }
        let opcode = (header >> 8) & 0xff;
        let register = (header >> 2) & 0x3f;
        result.packets.push(AgcPacket {
            offset: offset as u32,
            dwords,
            opcode,
            register,
        });

        if matches!(
            opcode,
            IT_DRAW_INDEX_INDIRECT
                | IT_DRAW_INDEX_2
                | IT_DRAW_INDEX_AUTO
                | IT_DRAW_INDEX_MULTI_AUTO
                | IT_DRAW_INDEX_OFFSET_2
                | IT_DRAW_INDEX_INDIRECT_MULTI
                | IT_DISPATCH_DRAW
        ) || (opcode == IT_NOP && matches!(register, R_DRAW_INDEX | R_DRAW_INDEX_AUTO))
        {
            result.draw_packets += 1;
        }
        if matches!(opcode, IT_DISPATCH_DIRECT | IT_DISPATCH_INDIRECT)
            || (opcode == IT_NOP && register == R_DISPATCH_DIRECT)
        {
            result.dispatch_packets += 1;
        }
        if opcode == IT_NOP && register == R_FLIP && dwords >= 6 {
            let body = &words[offset + 1..offset + dwords as usize];
            result.flips.push(AgcFlip {
                video_out_handle: body[0],
                display_buffer_index: body[1],
                flip_mode: body[2],
                flip_arg: u64::from(body[3]) | (u64::from(body[4]) << 32),
            });
        }
        if opcode == IT_EVENT_WRITE && dwords >= 2 {
            result.events.push(words[offset + 1] & 0x3f);
        }
        if opcode == IT_NOP && register == R_RELEASE_MEM && dwords >= 7 {
            let body = &words[offset + 1..offset + dwords as usize];
            let data_selection = (body[1] >> 16) & 0xff;
            let address = u64::from(body[2]) | (u64::from(body[3]) << 32);
            let value = u64::from(body[4]) | (u64::from(body[5]) << 32);
            let data = match data_selection {
                1 => body[4].to_le_bytes().to_vec(),
                2 | 3 => value.to_le_bytes().to_vec(),
                _ => Vec::new(),
            };
            if address != 0 && !data.is_empty() {
                result.memory_writes.push(AgcMemoryWrite {
                    packet_offset: offset as u32,
                    address,
                    data,
                });
            }
        }
        if opcode == IT_NOP && register == R_WRITE_DATA && dwords >= 4 {
            let body = &words[offset + 1..offset + dwords as usize];
            let control = body[0];
            let destination = control & 0xff;
            let increment = ((control >> 16) & 0xff) == 0;
            let address = u64::from(body[1]) | (u64::from(body[2]) << 32);
            if matches!(destination, 1 | 2 | 4 | 5) && address != 0 {
                for (index, value) in body[3..].iter().enumerate() {
                    result.memory_writes.push(AgcMemoryWrite {
                        packet_offset: offset as u32,
                        address: address + if increment { index as u64 * 4 } else { 0 },
                        data: value.to_le_bytes().to_vec(),
                    });
                }
            }
        }
        if opcode == IT_NOP && register == R_WAIT_MEM_32 && dwords >= 6 {
            let body = &words[offset + 1..offset + dwords as usize];
            result.waits32.push(AgcWait32 {
                packet_offset: offset as u32,
                address: u64::from(body[0]) | (u64::from(body[1]) << 32),
                mask: body[2],
                function: body[3],
                reference: body[4],
            });
        }
        offset += dwords as usize;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(dwords: u32, opcode: u32, register: u32) -> u32 {
        0xc000_0000 | ((dwords - 2) << 16) | (opcode << 8) | (register << 2)
    }

    #[test]
    fn decodes_draw_flip_dispatch_and_type2_packets() {
        let words = [
            header(2, IT_NOP, R_DRAW_INDEX_AUTO),
            3,
            header(2, IT_NOP, R_DRAW_INDEX),
            4,
            header(5, IT_DISPATCH_DIRECT, 0),
            2,
            1,
            1,
            0x41,
            header(6, IT_NOP, R_FLIP),
            7,
            2,
            1,
            0x89ab_cdef,
            0x0123_4567,
            header(2, IT_EVENT_WRITE, 0),
            0x2a,
            0x8000_0000,
        ];
        let decoded = decode_submission(&words).expect("valid AGC DCB");
        assert_eq!(decoded.packets.len(), 6);
        assert_eq!(decoded.draw_packets, 2);
        assert_eq!(decoded.dispatch_packets, 1);
        assert_eq!(decoded.events, [0x2a]);
        assert_eq!(
            decoded.flips,
            [AgcFlip {
                video_out_handle: 7,
                display_buffer_index: 2,
                flip_mode: 1,
                flip_arg: 0x0123_4567_89ab_cdef,
            }]
        );
    }

    #[test]
    fn rejects_truncated_and_non_type3_packets() {
        assert_eq!(
            decode_submission(&[header(8, IT_NOP, 0), 0]),
            Err(AgcDecodeError::Truncated {
                offset: 0,
                needed: 8,
                remaining: 2,
            })
        );
        assert_eq!(
            decode_submission(&[0]),
            Err(AgcDecodeError::UnsupportedPacketType {
                offset: 0,
                packet_type: 0,
            })
        );
    }

    #[test]
    fn decodes_release_mem_and_write_data_side_effects() {
        let words = [
            header(8, IT_NOP, R_RELEASE_MEM),
            0,
            2 << 16,
            0x1000,
            0,
            0x89ab_cdef,
            0x0123_4567,
            0,
            header(6, IT_NOP, R_WRITE_DATA),
            1,
            0x2000,
            0,
            0xaabb_ccdd,
            0x1122_3344,
            header(6, IT_NOP, R_WAIT_MEM_32),
            0x2000,
            0,
            0xffff_ffff,
            3,
            0xaabb_ccdd,
        ];
        let decoded = decode_submission(&words).unwrap();
        assert_eq!(
            decoded.memory_writes,
            [
                AgcMemoryWrite {
                    packet_offset: 0,
                    address: 0x1000,
                    data: 0x0123_4567_89ab_cdefu64.to_le_bytes().to_vec(),
                },
                AgcMemoryWrite {
                    packet_offset: 8,
                    address: 0x2000,
                    data: 0xaabb_ccddu32.to_le_bytes().to_vec(),
                },
                AgcMemoryWrite {
                    packet_offset: 8,
                    address: 0x2004,
                    data: 0x1122_3344u32.to_le_bytes().to_vec(),
                },
            ]
        );
        assert_eq!(
            decoded.waits32,
            [AgcWait32 {
                packet_offset: 14,
                address: 0x2000,
                mask: 0xffff_ffff,
                function: 3,
                reference: 0xaabb_ccdd,
            }]
        );
    }
}
