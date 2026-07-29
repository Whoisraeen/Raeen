//! Synthetic cost probe for the PM4 walk, per packet class.
//!
//! `RAEEN_TIME_DRAW` reports `walk_us` for a whole submission; this splits a
//! *synthetic* stream into one packet class at a time so a regression can be
//! attributed without the retail title. Run it explicitly:
//!
//! ```text
//! cargo test -p kyty-graphics --test pm4_walk_cost -- --ignored --nocapture
//! ```

use kyty_graphics::hw_regs::{Context, Shader, UserConfig};
use kyty_graphics::pm4;
use kyty_graphics::run::{CommandProcessor, DrawError, DrawSink, GuestMemory};

#[derive(Default)]
struct NullSink {
    draws: u64,
    boundaries: u64,
}

impl DrawSink for NullSink {
    fn guest_memory_write_boundary(&mut self, _writes: &[(u64, u64)]) {
        self.boundaries += 1;
    }

    fn draw_index_auto(
        &mut self,
        _ctx: &Context,
        _ucfg: &UserConfig,
        _sh: &Shader,
        _index_count: u32,
        _flags: u32,
    ) -> Result<(), DrawError> {
        self.draws += 1;
        Ok(())
    }

    fn dispatch_direct(
        &mut self,
        _ctx: &Context,
        _ucfg: &UserConfig,
        _sh: &Shader,
        _groups: [u32; 3],
        _mode: u32,
    ) -> Result<(), DrawError> {
        Ok(())
    }
}

struct Scratch {
    base: u64,
    words: std::cell::RefCell<Vec<u32>>,
}

impl GuestMemory for Scratch {
    fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
        let rel = addr.checked_sub(self.base)?;
        if rel % 4 != 0 {
            return None;
        }
        let start = usize::try_from(rel / 4).ok()?;
        let end = start.checked_add(count as usize)?;
        self.words.borrow().get(start..end).map(<[u32]>::to_vec)
    }

    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        let Some(rel) = addr.checked_sub(self.base) else {
            return false;
        };
        let Ok(start) = usize::try_from(rel) else {
            return false;
        };
        let mut words = self.words.borrow_mut();
        // SAFETY-free byte view over the dword scratch: tests only.
        let len = words.len() * 4;
        if start + bytes.len() > len {
            return false;
        }
        for (i, &b) in bytes.iter().enumerate() {
            let off = start + i;
            let w = &mut words[off / 4];
            let shift = (off % 4) * 8;
            *w = (*w & !(0xffu32 << shift)) | (u32::from(b) << shift);
        }
        true
    }
}

fn time<T>(label: &str, iters: u64, f: impl FnOnce() -> T) -> T {
    let at = std::time::Instant::now();
    let out = f();
    let ns = at.elapsed().as_nanos() as u64;
    println!(
        "{label:<34} total={:>9.3} ms   per-item={:>8.1} ns",
        ns as f64 / 1e6,
        ns as f64 / iters as f64
    );
    out
}

/// One `SET_CONTEXT_REG` per register, one register per packet — the shape a
/// title's per-draw state block has.
fn context_reg_stream(packets: usize) -> Vec<u32> {
    let mut dcb = Vec::with_capacity(packets * 3);
    for i in 0..packets {
        dcb.push(pm4::header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO));
        // Rotate over the registers a real stream programs most: blend, colour
        // info, viewport, and the shader-facing SPI block.
        dcb.push(pm4::SPI_PS_INPUT_CNTL_0 + (i as u32 % 32));
        dcb.push(0x1234_5678);
    }
    dcb
}

/// User-SGPR writes: 16 registers in one packet, the AGC per-draw constant push.
fn sh_reg_stream(packets: usize) -> Vec<u32> {
    let mut dcb = Vec::with_capacity(packets * 18);
    for i in 0..packets {
        dcb.push(pm4::header(18, pm4::IT_SET_SH_REG, pm4::R_ZERO));
        dcb.push(pm4::SPI_SHADER_USER_DATA_PS_0);
        for j in 0..16u32 {
            dcb.push(0x1000_0000 + i as u32 * 16 + j);
        }
    }
    dcb
}

fn main_probe(label: &str, dcb: &[u32], packets: usize, mem: Option<&dyn GuestMemory>) {
    let mut cp = CommandProcessor::new();
    let mut sink = NullSink::default();
    // Warm.
    cp.run_with_memory(dcb, &mut sink, mem).expect("warm walk");
    let reps = 20u64;
    time(label, reps * packets as u64, || {
        for _ in 0..reps {
            cp.run_with_memory(dcb, &mut sink, mem).expect("walk");
        }
    });
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn pm4_walk_cost_by_packet_class() {
    const PACKETS: usize = 20_000;

    let ctx = context_reg_stream(PACKETS);
    main_probe("SET_CONTEXT_REG (1 reg/packet)", &ctx, PACKETS, None);

    let sh = sh_reg_stream(PACKETS);
    main_probe("SET_SH_REG (16 regs/packet)", &sh, PACKETS * 16, None);

    // NOPs: the walk's floor.
    let mut nops = Vec::with_capacity(PACKETS * 3);
    for _ in 0..PACKETS {
        nops.push(pm4::header(3, pm4::IT_NOP, pm4::R_ZERO));
        nops.push(0);
        nops.push(0);
    }
    main_probe("IT_NOP (walk floor)", &nops, PACKETS, None);

    // WRITE_DATA completion labels: the write-boundary notification path.
    let scratch = Scratch {
        base: 0x1_0000_0000,
        words: std::cell::RefCell::new(vec![0; 4096]),
    };
    let mut labels = Vec::with_capacity(PACKETS * 6);
    for i in 0..PACKETS {
        labels.push(pm4::header(6, pm4::IT_WRITE_DATA, pm4::R_ZERO));
        labels.push(1 << 8); // dst_sel = memory (see cp_op_write_data)
        let addr = scratch.base + ((i as u64 % 512) * 8);
        labels.push(addr as u32);
        labels.push((addr >> 32) as u32);
        labels.push(0xabcd_0000 + i as u32);
        labels.push(0);
    }
    main_probe(
        "IT_WRITE_DATA (label + boundary)",
        &labels,
        PACKETS,
        Some(&scratch),
    );
}
