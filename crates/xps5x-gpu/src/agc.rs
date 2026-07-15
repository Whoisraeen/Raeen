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
const IT_DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_DISPATCH_DRAW: u32 = 0x8d;

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

/// Structural facts extracted from a complete DCB submission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgcSubmission {
    pub packets: Vec<AgcPacket>,
    pub draw_packets: u32,
    pub dispatch_packets: u32,
    pub flips: Vec<AgcFlip>,
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
            0x8000_0000,
        ];
        let decoded = decode_submission(&words).expect("valid AGC DCB");
        assert_eq!(decoded.packets.len(), 5);
        assert_eq!(decoded.draw_packets, 2);
        assert_eq!(decoded.dispatch_packets, 1);
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
}
