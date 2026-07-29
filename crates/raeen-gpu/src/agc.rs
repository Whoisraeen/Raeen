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
/// The NON-indexed indirect draws, `IT_DRAW_INDIRECT` / `IT_DRAW_INDIRECT_MULTI`.
///
/// Their indexed twins (0x25 / 0x38) were counted here from the start and these
/// two were not, even though `CommandProcessor::dispatch` routes all four to the
/// same `cp_op_draw_indirect` and issues a real `DrawSink` draw for each. Found
/// by the reverse-direction half of `tests/pm4_decoder_agreement.rs` while
/// closing the `IT_DISPATCH_DRAW_PREAMBLE` gap — the same under-count, and
/// invisible for exactly the same reason: nothing compared the two decoders'
/// opcode sets.
const IT_DRAW_INDIRECT: u32 = 0x24;
const IT_DRAW_INDIRECT_MULTI: u32 = 0x2c;
const R_DRAW_INDEX_AUTO: u32 = 0x04;
const R_DRAW_INDEX: u32 = 0x03;
const R_FLIP: u32 = 0x17;
const R_DISPATCH_DIRECT: u32 = 0x08;
const R_WRITE_DATA: u32 = 0x15;
const R_RELEASE_MEM: u32 = 0x18;
const R_WAIT_MEM_32: u32 = 0x0a;
const IT_DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
/// The AGC multi-instanced indexed draw (KytyPS5 `pm4.h` L44,
/// `kyty_graphics::pm4::IT_DISPATCH_DRAW_PREAMBLE`).
///
/// Raeen's own `sceAgcDcbDrawIndexMultiInstanced` emits this, so leaving it out
/// of the draw match UNDER-reported `draw_packets` for any title that calls it —
/// the mirror image of the 0x30/0x8d over-report, and the reason
/// `raeen-gpu/tests/pm4_decoder_agreement.rs` now checks both directions.
const IT_DISPATCH_DRAW_PREAMBLE: u32 = 0x3a;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_DISPATCH_DRAW: u32 = 0x8d;
const IT_EVENT_WRITE: u32 = 0x46;
const IT_WRITE_DATA: u32 = 0x37;
const IT_DMA_DATA: u32 = 0x50;

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

/// One end-of-pipe interrupt requested by a `RELEASE_MEM` packet (`interrupt`
/// field, body DWORD 1 bits 31:24). Hardware raises the EOP interrupt the
/// kernel delivers to events registered via `sceAgcDriverAddEqEvent`; the
/// decoder is pure, so it only reports the request — the submit layer triggers
/// the registered events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcEopInterrupt {
    /// DWORD offset of the packet that requested the interrupt.
    pub packet_offset: u32,
    pub interrupt: u32,
    /// The packet's trailing interrupt-context DWORD.
    pub context_id: u32,
}

/// One GPU-timestamp label write requested by a `RELEASE_MEM` packet with
/// `data_selection` 3. Hardware writes the GPU core clock counter — non-zero
/// and monotonic; the packet's immediate data field is meaningless for this
/// selection. The decoder reports the target address and the submit layer
/// supplies the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcTimestampWrite {
    /// DWORD offset of the packet that requested the timestamp.
    pub packet_offset: u32,
    pub address: u64,
}

/// One guest-memory copy requested by an `IT_DMA_DATA` packet (Memory →
/// Memory). The submit layer performs the copy — the decoder is pure and
/// holds no guest-memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcMemoryCopy {
    /// DWORD offset of the packet that produced this copy.
    pub packet_offset: u32,
    pub src: u64,
    pub dst: u64,
    pub num_bytes: u32,
}

/// One 32-bit-pattern fill requested by an `IT_DMA_DATA` packet
/// (Data → Memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgcMemoryFill {
    /// DWORD offset of the packet that produced this fill.
    pub packet_offset: u32,
    pub address: u64,
    pub value: u32,
    pub num_bytes: u32,
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
    pub memory_copies: Vec<AgcMemoryCopy>,
    pub memory_fills: Vec<AgcMemoryFill>,
    pub waits32: Vec<AgcWait32>,
    /// Event ids signaled by standard `EVENT_WRITE` packets.
    pub events: Vec<u32>,
    /// End-of-pipe interrupts requested by `RELEASE_MEM` packets.
    pub eop_interrupts: Vec<AgcEopInterrupt>,
    /// GPU-timestamp label writes (`RELEASE_MEM` `data_selection` 3).
    pub timestamp_writes: Vec<AgcTimestampWrite>,
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
                | IT_DRAW_INDIRECT
                | IT_DRAW_INDIRECT_MULTI
                | IT_DRAW_INDEX_2
                | IT_DRAW_INDEX_AUTO
                | IT_DRAW_INDEX_MULTI_AUTO
                | IT_DRAW_INDEX_OFFSET_2
                | IT_DRAW_INDEX_INDIRECT_MULTI
                | IT_DISPATCH_DRAW_PREAMBLE
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
            let interrupt = (body[1] >> 24) & 0xff;
            let address = u64::from(body[2]) | (u64::from(body[3]) << 32);
            let value = u64::from(body[4]) | (u64::from(body[5]) << 32);
            let data = match data_selection {
                1 => body[4].to_le_bytes().to_vec(),
                2 => value.to_le_bytes().to_vec(),
                _ => Vec::new(),
            };
            if address != 0 && !data.is_empty() {
                result.memory_writes.push(AgcMemoryWrite {
                    packet_offset: offset as u32,
                    address,
                    data,
                });
            }
            // data_selection 3 asks for the GPU core clock counter; the
            // packet's immediate is zero in real streams, so it must not be
            // written verbatim (a title polling the label for non-zero would
            // wait forever).
            if data_selection == 3 && address != 0 {
                result.timestamp_writes.push(AgcTimestampWrite {
                    packet_offset: offset as u32,
                    address,
                });
            }
            if interrupt != 0 {
                result.eop_interrupts.push(AgcEopInterrupt {
                    packet_offset: offset as u32,
                    interrupt,
                    context_id: body.get(6).copied().unwrap_or(0),
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
        // Standard PM4 `IT_WRITE_DATA` (0x37): control, dst lo/hi, then data
        // dwords. shadPS4's liverpool asserts dst_sel 2 (TCL2) or 5 (immediate
        // memory); 1 (Memory) is accepted for robustness — all three mean the
        // dwords land in guest memory at the destination address.
        if opcode == IT_WRITE_DATA && dwords >= 5 {
            let body = &words[offset + 1..offset + dwords as usize];
            let control = body[0];
            let dst_sel = (control >> 8) & 0xf;
            let wr_one_addr = (control >> 16) & 1 != 0;
            let address = u64::from(body[1]) | (u64::from(body[2]) << 32);
            if matches!(dst_sel, 1 | 2 | 5) && address != 0 {
                if wr_one_addr {
                    // Every dword targets the same address (e.g. register
                    // programming): one write per dword, no increment.
                    for value in &body[3..] {
                        result.memory_writes.push(AgcMemoryWrite {
                            packet_offset: offset as u32,
                            address,
                            data: value.to_le_bytes().to_vec(),
                        });
                    }
                } else {
                    let mut data = Vec::with_capacity((body.len() - 3) * 4);
                    for value in &body[3..] {
                        data.extend_from_slice(&value.to_le_bytes());
                    }
                    result.memory_writes.push(AgcMemoryWrite {
                        packet_offset: offset as u32,
                        address,
                        data,
                    });
                }
            }
        }
        // Standard PM4 `IT_DMA_DATA` (0x50): control, src lo/hi (or the fill
        // value for src_sel=Data), dst lo/hi, command (low 21 bits = bytes).
        // Only the plain guest-memory cases are modeled; GDS transfers have
        // no model and are consumed silently like the other sync ops.
        if opcode == IT_DMA_DATA && dwords >= 7 {
            let body = &words[offset + 1..offset + dwords as usize];
            let control = body[0];
            let dst_sel = (control >> 20) & 0x3;
            let src_sel = (control >> 29) & 0x3;
            let num_bytes = body[5] & 0x1f_ffff;
            let src = u64::from(body[1]) | (u64::from(body[2]) << 32);
            let dst = u64::from(body[3]) | (u64::from(body[4]) << 32);
            // 0 = Memory, 3 = MemoryUsingL2 — both are guest memory here.
            match (src_sel, dst_sel) {
                (0 | 3, 0 | 3) if src != 0 && dst != 0 && num_bytes != 0 => {
                    result.memory_copies.push(AgcMemoryCopy {
                        packet_offset: offset as u32,
                        src,
                        dst,
                        num_bytes,
                    });
                }
                (2, 0 | 3) if dst != 0 && num_bytes != 0 => {
                    result.memory_fills.push(AgcMemoryFill {
                        packet_offset: offset as u32,
                        address: dst,
                        value: body[1],
                        num_bytes,
                    });
                }
                _ => {}
            }
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

    #[test]
    fn release_mem_timestamp_selection_reports_address_not_zero_immediate() {
        // data_selection 3 = GPU clock counter. Real streams leave the packet's
        // immediate zero (hardware supplies the clock), so decoding it as an
        // immediate write left titles polling a fence stuck at zero.
        let words = [
            header(8, IT_NOP, R_RELEASE_MEM),
            0,
            3 << 16,
            0x3000,
            0,
            0,
            0,
            0,
        ];
        let decoded = decode_submission(&words).unwrap();
        assert!(decoded.memory_writes.is_empty());
        assert_eq!(
            decoded.timestamp_writes,
            [AgcTimestampWrite {
                packet_offset: 0,
                address: 0x3000,
            }]
        );
        assert!(decoded.eop_interrupts.is_empty());
    }

    #[test]
    fn release_mem_interrupt_forms_report_eop_interrupts() {
        // Interrupt-only (data_selection 0, no address) previously decoded to
        // nothing at all; an interrupt riding on a sel=1 write must report
        // both the write and the interrupt.
        let words = [
            header(8, IT_NOP, R_RELEASE_MEM),
            0,
            2 << 24,
            0,
            0,
            0,
            0,
            0x77,
            header(8, IT_NOP, R_RELEASE_MEM),
            0,
            (1 << 16) | (3 << 24),
            0x4000,
            0,
            0x5555_5555,
            0,
            0x88,
        ];
        let decoded = decode_submission(&words).unwrap();
        assert_eq!(
            decoded.eop_interrupts,
            [
                AgcEopInterrupt {
                    packet_offset: 0,
                    interrupt: 2,
                    context_id: 0x77,
                },
                AgcEopInterrupt {
                    packet_offset: 8,
                    interrupt: 3,
                    context_id: 0x88,
                },
            ]
        );
        assert_eq!(
            decoded.memory_writes,
            [AgcMemoryWrite {
                packet_offset: 8,
                address: 0x4000,
                data: 0x5555_5555u32.to_le_bytes().to_vec(),
            }]
        );
    }

    #[test]
    fn decodes_it_write_data_and_it_dma_data_side_effects() {
        let words = [
            // IT_WRITE_DATA: dst_sel=2 (TCL2), incrementing, 2 data dwords.
            header(6, IT_WRITE_DATA, 0),
            2 << 8,
            0x5000,
            0,
            0xdead_beef,
            0xcafe_f00d,
            // IT_DMA_DATA Memory→Memory: src 0x6000 → dst 0x9000, 256 bytes.
            header(7, IT_DMA_DATA, 0),
            0, // src_sel=0 (Memory), dst_sel=0 (Memory)
            0x6000,
            0,
            0x9000,
            0,
            256,
            // IT_DMA_DATA Data→Memory: fill dst 0xa000 with 0x1122_3344, 64 bytes.
            header(7, IT_DMA_DATA, 0),
            2 << 29, // src_sel=2 (Data)
            0x1122_3344,
            0,
            0xa000,
            0,
            64,
            // IT_DMA_DATA with a GDS endpoint: no model — no record at all.
            header(7, IT_DMA_DATA, 0),
            1 << 20, // dst_sel=1 (Gds)
            0x6000,
            0,
            0x9000,
            0,
            256,
        ];
        let decoded = decode_submission(&words).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&0xdead_beefu32.to_le_bytes());
        expected.extend_from_slice(&0xcafe_f00du32.to_le_bytes());
        assert_eq!(
            decoded.memory_writes,
            [AgcMemoryWrite {
                packet_offset: 0,
                address: 0x5000,
                data: expected,
            }]
        );
        assert_eq!(
            decoded.memory_copies,
            [AgcMemoryCopy {
                packet_offset: 6,
                src: 0x6000,
                dst: 0x9000,
                num_bytes: 256,
            }]
        );
        assert_eq!(
            decoded.memory_fills,
            [AgcMemoryFill {
                packet_offset: 13,
                address: 0xa000,
                value: 0x1122_3344,
                num_bytes: 64,
            }]
        );
        // wr_one_addr: same address for every dword.
        let one_addr = [
            header(6, IT_WRITE_DATA, 0),
            (2 << 8) | (1 << 16),
            0x7000,
            0,
            0x1111_1111,
            0x2222_2222,
        ];
        let decoded = decode_submission(&one_addr).unwrap();
        assert_eq!(
            decoded.memory_writes,
            [
                AgcMemoryWrite {
                    packet_offset: 0,
                    address: 0x7000,
                    data: 0x1111_1111u32.to_le_bytes().to_vec(),
                },
                AgcMemoryWrite {
                    packet_offset: 0,
                    address: 0x7000,
                    data: 0x2222_2222u32.to_le_bytes().to_vec(),
                },
            ]
        );
    }
}
