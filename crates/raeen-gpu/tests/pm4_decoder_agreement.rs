//! The two PM4 decoders must agree on what a draw is.
//!
//! Raeen walks a Gen5 command buffer twice with two independent decoders:
//!
//! - [`raeen_gpu::agc::decode_submission`] — the eager, structural pass. Its
//!   `draw_packets` / `dispatch_packets` feed `ctx.kernel.agc_draw_packet_count`
//!   (`raeen-hle/src/libsce_agc.rs`), i.e. the number a session report and every
//!   "the title asked for N draws" diagnostic is built on.
//! - [`kyty_graphics::run::CommandProcessor`] — the executing pass, which
//!   translates packets into [`DrawSink`] calls.
//!
//! When the first counts an opcode the second has no arm for, the packet
//! inflates the submission's draw count while the walk records **nothing**: not
//! `sink.draws`, not `sink.draw_skips`, and not
//! [`CommandProcessor::refused_draws`]. The only trace is one rate-limited
//! `warn!` per distinct opcode per processor instance. That is precisely the
//! shape of the Dead Cells `draws=0` blocker — a draw counted by one decoder and
//! dropped by the other without a counted reason — and it stayed invisible
//! because nothing tied the two opcode sets together.
//!
//! This is that tie. The opcode set is **discovered from `decode_submission` at
//! run time**, never listed here, so a new draw opcode added to one decoder and
//! not the other fails this test instead of silently costing a frame.
//!
//! # Both directions
//!
//! The first version of this file checked one direction only — counted implies
//! handled — and `IT_DISPATCH_DRAW_PREAMBLE` (0x3A) promptly turned up drifting
//! the OTHER way: `decode_submission` did not count it, the command processor
//! had no arm, and Raeen's own `sceAgcDcbDrawIndexMultiInstanced` emitted it.
//! The submission UNDER-reported its draws and the draw vanished. So the
//! invariant is now symmetric:
//!
//! - counted by `decode_submission` ⇒ the command processor must draw it or
//!   refuse it by name ([`every_agc_counted_draw_opcode_is_accounted_for_by_the_command_processor`]);
//! - drawn by the command processor ⇒ `decode_submission` must count it
//!   ([`every_opcode_the_command_processor_draws_is_counted_by_the_agc_decoder`]).
//!
//! Neither direction can see Raeen's *emitters*, though — a draw opcode
//! `raeen-hle` writes that neither decoder knows would still pass both. That
//! third edge is pinned from the emitter's side, in `raeen-hle`'s
//! `multi_instanced_draw_emission_reaches_the_command_processor`.
//!
//! Synthetic PM4 only — every fixture is built from [`kyty_graphics::pm4`]
//! constants. No retail content.

use kyty_graphics::hw_regs::{Context, Shader, UserConfig};
use kyty_graphics::pm4::{self, ItOp, RCode};
use kyty_graphics::run::{CommandProcessor, DrawError, DrawSink, GuestMemory, IndexedDraw};

/// Body-dword lengths each candidate opcode is probed at.
///
/// The packet layouts differ per opcode and this test deliberately encodes
/// none of them — knowing them is the *handler's* job, and a table here would
/// be one more thing to drift. Instead an opcode passes if **any** well-formed
/// length reaches the sink or is refused; an opcode with no arm at all fails at
/// every length, because the default arm returns `Ok(body_dw)` without touching
/// the sink or the refusal counter.
///
/// The range covers every draw/dispatch body in the processor today: the
/// longest is the AGC `R_DRAW_INDEX` form at 5, and `IT_DISPATCH_INDIRECT`
/// discriminates its two encodings at 2 and 3.
const PROBE_BODY_DWORDS: std::ops::RangeInclusive<u32> = 1..=10;

/// Every address readable, every dword non-zero.
///
/// Indirect draws recover their vertex count from the first args record and
/// return early when it is zero, so a zero-filled reader would make a
/// correctly-handled opcode look unhandled.
struct AnyMemory;

impl GuestMemory for AnyMemory {
    fn read_dwords(&self, _addr: u64, count: u32) -> Option<Vec<u32>> {
        Some(vec![3; count as usize])
    }
}

#[derive(Default)]
struct CountingSink {
    draws: u32,
    dispatches: u32,
}

impl CountingSink {
    const fn total(&self) -> u32 {
        self.draws + self.dispatches
    }
}

impl DrawSink for CountingSink {
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

    /// Overridden so an indexed draw is counted as itself rather than through
    /// the trait's degrade-to-`draw_index_auto` default. Either would satisfy
    /// the invariant; counting it here keeps the diagnostic honest about which
    /// entry point the packet actually reached.
    fn draw_index(
        &mut self,
        _ctx: &Context,
        _ucfg: &UserConfig,
        _sh: &Shader,
        _draw: &IndexedDraw,
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
        self.dispatches += 1;
        Ok(())
    }
}

/// One well-formed type-3 packet: `body_dw` body dwords, every one non-zero so
/// no handler mistakes the fixture for an unprogrammed address or a zero count.
fn probe_packet(op: u8, register: u8, body_dw: u32) -> Vec<u32> {
    let total_dw = u16::try_from(body_dw + 1).expect("probe lengths are tiny");
    let mut words = vec![pm4::header(total_dw, ItOp(op), RCode(register))];
    words.extend(std::iter::repeat_n(1u32, body_dw as usize));
    words
}

/// GPU state a draw needs before it can do anything: the indirect-argument
/// bases (draw and dispatch), a bound index buffer, and an index type.
///
/// Without these, correctly-handled indirect opcodes skip themselves with a
/// warn and would read as unhandled. Nothing here draws, dispatches, or
/// refuses — [`prologue_is_inert`] pins that.
fn prologue() -> Vec<u32> {
    vec![
        // IT_SET_BASE select 1, shader type 0 → indirect DRAW args base.
        pm4::header(4, pm4::IT_SET_BASE, pm4::R_ZERO),
        1,
        0x1000,
        0,
        // The same, with PM4 header bit 1 (`Gnmp::ShaderType`) set → indirect
        // DISPATCH args base. KytyPS5 `CpOpSetBase`, pm4Handlers.cpp L2546.
        pm4::header(4, pm4::IT_SET_BASE, pm4::R_ZERO) | 0x2,
        1,
        0x2000,
        0,
        // A bound index buffer, so IT_DRAW_INDEX_OFFSET_2 has somewhere to
        // point.
        pm4::header(3, pm4::IT_INDEX_BASE, pm4::R_ZERO),
        0x3000,
        0,
        // 32-bit indices.
        pm4::header(2, pm4::IT_INDEX_TYPE, pm4::R_ZERO),
        1,
    ]
}

/// What the command processor did with one probe packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Accounting {
    sink_calls: u32,
    refused: u64,
}

impl Accounting {
    /// Whether the packet was accounted for at all — translated into sink work,
    /// or refused by name and counted. Anything else means it fell to the
    /// anonymous unknown-opcode arm, which is the drift this test exists to
    /// catch.
    const fn is_accountable(self) -> bool {
        self.sink_calls > 0 || self.refused > 0
    }
}

/// Run `prologue() ++ probe_packet(..)` through a fresh command processor.
fn walk_probe(op: u8, register: u8, body_dw: u32) -> Accounting {
    let mut dcb = prologue();
    dcb.extend(probe_packet(op, register, body_dw));

    let mut cp = CommandProcessor::new();
    let mut sink = CountingSink::default();
    // A structural fault (`Truncated` / `NotType3`) is not "accounted for"
    // either, so the result is deliberately ignored: what matters is whether
    // the sink or the refusal counter moved.
    let _ = cp.run_with_memory(&dcb, &mut sink, Some(&AnyMemory));

    Accounting {
        sink_calls: sink.total(),
        refused: cp.refused_draws(),
    }
}

/// Does `decode_submission` count this packet toward `draw_packets` or
/// `dispatch_packets`?
fn agc_counts_as_draw_or_dispatch(op: u8, register: u8) -> bool {
    let words = probe_packet(op, register, 1);
    let decoded = raeen_gpu::agc::decode_submission(&words)
        .expect("one well-formed type-3 packet must decode");
    decoded.draw_packets + decoded.dispatch_packets > 0
}

/// Every `(opcode, register)` pair the AGC decoder counts as a draw or a
/// dispatch.
///
/// Brute-forced over the whole encodable space rather than read from a list:
/// the AGC dialect discriminates most of its operations on the register field
/// of `IT_NOP` (bits 7:2) and standard PM4 on the opcode (bits 15:8), and a
/// future decoder change could key off either. Sweeping both means this test
/// cannot go stale on a field it did not anticipate.
fn counted_draw_opcodes() -> Vec<(u8, u8)> {
    let mut found = Vec::new();
    for op in 0u8..=0xff {
        for register in 0u8..=0x3f {
            if agc_counts_as_draw_or_dispatch(op, register) {
                found.push((op, register));
            }
        }
    }
    found
}

/// **The invariant.** Every opcode `decode_submission` counts as a draw or a
/// dispatch must have a non-default arm in `CommandProcessor::dispatch` — a
/// real handler, or an explicit named+counted refusal.
///
/// The failure this prevents: a packet that raises
/// `ctx.kernel.agc_draw_packet_count` while the command processor increments
/// neither `sink.draws`, nor `sink.draw_skips`, nor `refused_draws`, leaving a
/// single rate-limited warn as the only evidence that a frame's geometry went
/// missing.
#[test]
fn every_agc_counted_draw_opcode_is_accounted_for_by_the_command_processor() {
    let mut drifted = Vec::new();

    for (op, register) in counted_draw_opcodes() {
        let accounted = PROBE_BODY_DWORDS
            .clone()
            .any(|body_dw| walk_probe(op, register, body_dw).is_accountable());
        if !accounted {
            drifted.push((op, register));
        }
    }

    assert!(
        drifted.is_empty(),
        "PM4 decoder drift — agc::decode_submission counts these as draws or \
         dispatches, but kyty_graphics::run::CommandProcessor has no arm for them at \
         ANY body length in {PROBE_BODY_DWORDS:?}:\n  - {}\n\
         Such a packet inflates ctx.kernel.agc_draw_packet_count while the walk records \
         neither a draw, nor a draw_skip, nor a refused_draw — the Dead Cells `draws=0` \
         failure shape. Fix by adding a handler to `CommandProcessor::dispatch`, or an \
         explicit named+counted refusal arm so the drop lands in refused_draws / \
         last_refusal (see the IT_DRAW_INDEX_MULTI_AUTO / IT_DISPATCH_DRAW arm).",
        describe_drift(&drifted)
    );
}

/// **The reverse invariant.** Every opcode the command processor actually
/// translates into a draw or dispatch must be counted by `decode_submission`.
///
/// This is the direction `IT_DISPATCH_DRAW_PREAMBLE` (0x3A) drifted in: a real
/// draw that the eager decoder did not count, so
/// `ctx.kernel.agc_draw_packet_count` UNDER-reported the frame. Under-reporting
/// is the more insidious half — an inflated count at least shows up as a
/// mismatch against `draws`, while a missing one makes a dropped draw look like
/// a draw that was never requested.
///
/// Only sink calls count here, not refusals: a refused draw means the processor
/// recognized the opcode but could not translate it, which says nothing about
/// whether it is a draw the eager decoder should count.
#[test]
fn every_opcode_the_command_processor_draws_is_counted_by_the_agc_decoder() {
    let mut drifted = Vec::new();

    for op in 0u8..=0xff {
        for register in 0u8..=0x3f {
            let draws = PROBE_BODY_DWORDS
                .clone()
                .any(|body_dw| walk_probe(op, register, body_dw).sink_calls > 0);
            if draws && !agc_counts_as_draw_or_dispatch(op, register) {
                drifted.push((op, register));
            }
        }
    }

    assert!(
        drifted.is_empty(),
        "PM4 decoder drift (reverse direction) — kyty_graphics::run::CommandProcessor \
         translates these into real DrawSink draws/dispatches, but \
         agc::decode_submission counts none of them:\n  - {}\n\
         ctx.kernel.agc_draw_packet_count therefore UNDER-reports the submission, so a \
         draw that is later dropped reads as a draw the title never asked for. Fix by \
         adding the opcode to the draw/dispatch match in `crates/raeen-gpu/src/agc.rs`.",
        describe_drift(&drifted)
    );
}

/// Render drifted `(opcode, register)` pairs one line per opcode.
///
/// Most PM4 opcodes are counted regardless of the register field, so a raw
/// listing repeats the same opcode 64 times and buries the one fact the reader
/// needs. Collapse those into `(any register)` and spell the registers out only
/// when the opcode drifts for a subset — which is what an `IT_NOP`-dialect
/// regression would look like.
fn describe_drift(drifted: &[(u8, u8)]) -> String {
    const ALL_REGISTERS: usize = 0x40;
    let mut by_opcode: std::collections::BTreeMap<u8, Vec<u8>> = std::collections::BTreeMap::new();
    for (op, register) in drifted {
        by_opcode.entry(*op).or_default().push(*register);
    }
    by_opcode
        .into_iter()
        .map(|(op, registers)| {
            if registers.len() == ALL_REGISTERS {
                format!("opcode {op:#04x} (any register)")
            } else {
                let list = registers
                    .iter()
                    .map(|r| format!("{r:#04x}"))
                    .collect::<Vec<_>>()
                    .join("/");
                format!("opcode {op:#04x} register {list}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n  - ")
}

/// Anti-vacuity: the sweep must actually discover the opcodes we know are
/// counted. A probe harness that silently found nothing would make the
/// invariant above pass forever.
#[test]
fn the_sweep_discovers_the_known_counted_draw_opcodes() {
    let found = counted_draw_opcodes();
    let opcodes: std::collections::BTreeSet<u8> = found.iter().map(|(op, _)| *op).collect();

    for (op, name) in [
        (0x15u8, "IT_DISPATCH_DIRECT"),
        (0x16, "IT_DISPATCH_INDIRECT"),
        (0x24, "IT_DRAW_INDIRECT"),
        (0x25, "IT_DRAW_INDEX_INDIRECT"),
        (0x27, "IT_DRAW_INDEX_2"),
        (0x2c, "IT_DRAW_INDIRECT_MULTI"),
        (0x2d, "IT_DRAW_INDEX_AUTO"),
        (0x30, "IT_DRAW_INDEX_MULTI_AUTO"),
        (0x35, "IT_DRAW_INDEX_OFFSET_2"),
        (0x3a, "IT_DISPATCH_DRAW_PREAMBLE"),
        (0x38, "IT_DRAW_INDEX_INDIRECT_MULTI"),
        (0x8d, "IT_DISPATCH_DRAW"),
    ] {
        assert!(
            opcodes.contains(&op),
            "the sweep must find {name} ({op:#04x}) — if decode_submission genuinely \
             stopped counting it, delete this row; otherwise the probe harness is broken"
        );
    }

    // The AGC dialect's NOP-wrapped draw/dispatch forms.
    for (register, name) in [
        (0x03u8, "R_DRAW_INDEX"),
        (0x04, "R_DRAW_INDEX_AUTO"),
        (0x08, "R_DISPATCH_DIRECT"),
    ] {
        assert!(
            found.contains(&(pm4::IT_NOP.0, register)),
            "the sweep must find IT_NOP + {name} ({register:#04x})"
        );
    }
}

/// The probe harness itself must be inert: if the prologue drew, dispatched, or
/// refused anything, every opcode would look accounted for and the invariant
/// would be meaningless.
#[test]
fn prologue_is_inert() {
    let mut cp = CommandProcessor::new();
    let mut sink = CountingSink::default();
    cp.run_with_memory(&prologue(), &mut sink, Some(&AnyMemory))
        .expect("the prologue is well-formed PM4");

    assert_eq!(sink.total(), 0, "the prologue must not reach the sink");
    assert_eq!(
        cp.refused_draws(),
        0,
        "the prologue must not refuse anything"
    );
    assert_ne!(
        cp.indirect_draw_base(),
        0,
        "the prologue must actually program the indirect-draw base, or the \
         indirect opcodes skip themselves and read as unhandled"
    );
    assert_ne!(cp.index_base(), 0, "and the index base");
}

/// A control: an opcode `decode_submission` does NOT count as a draw and the
/// command processor does NOT handle falls to the anonymous arm — unaccounted.
///
/// This proves [`Accounting::is_accountable`] can return `false`, i.e. that the
/// invariant test is capable of failing at all.
#[test]
fn an_unhandled_opcode_is_detectably_unaccounted() {
    // 0x0e is not a PM4 opcode Raeen decodes on either side.
    const UNHANDLED: u8 = 0x0e;
    assert!(
        !agc_counts_as_draw_or_dispatch(UNHANDLED, 0),
        "picked a control opcode that IS counted — choose another"
    );
    for body_dw in PROBE_BODY_DWORDS {
        assert!(
            !walk_probe(UNHANDLED, 0, body_dw).is_accountable(),
            "an opcode with no arm must leave the sink and refusal counter untouched, \
             or `is_accountable` cannot distinguish drift from a handled packet"
        );
    }
}
