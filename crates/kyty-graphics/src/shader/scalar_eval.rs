//! Compile-time evaluator for the scalar (SGPR) instruction stream.
//!
//! Ported from SharpEmu's `Gen5ShaderScalarEvaluator.cs`
//! (`src/SharpEmu.ShaderCompiler/`, GPL-2.0, © SharpEmu contributors) —
//! `TryExecuteScalarAlu` / `TryExecuteScalarCompare` / `TryEvaluateScalarOperand`
//! supply the per-opcode semantics; individual op semantics cross-checked
//! against KytyPS5 (MIT, © InoriRus / Nmzik). See `THIRD_PARTY_NOTICES.md`.
//!
//! # Why this exists
//!
//! An RDNA2 scalar memory load addresses `base_pair + SGPR[soffset] + imm`. The
//! recompiler can only lower such a load when the *whole* address is a
//! dispatch-time constant, so the soffset register's value has to be proven at
//! translate time. Before this module the only provable shapes were
//!
//! * a live-in user-data register nothing writes, and
//! * a single preceding `s_mov_b32` / `s_movk_i32` from a literal,
//!
//! which turned every computed offset (`s_lshl_b32`, `s_add_u32`,
//! `s_and_b32`, `s_mul_i32`, `s_bfe_u32`, `s_cselect_b32`, …) into the named
//! `unresolved register soffset` refusal — a skipped draw/dispatch and missing
//! geometry. This module folds the arithmetic instead.
//!
//! # The soundness asymmetry (read before relaxing anything here)
//!
//! The two failure modes are **not** symmetric:
//!
//! * saying **unknown** when the value was in fact knowable costs one skipped
//!   dispatch — exactly the behaviour that already exists today;
//! * saying **`Known(x)`** when the real value is not `x` makes the caller read
//!   the wrong descriptor dwords out of guest memory and hand a bogus V#/T#/S#
//!   to Vulkan. That class has already cost this project a measured
//!   `VK_ERROR_DEVICE_LOST` (see the EUD-pointer comment in
//!   `shader_capture_runtime_scalar_loads_shifted`).
//!
//! Therefore every operation here is *total*: an unrepresentable operand, an
//! unmodelled opcode, a value that depends on guest memory, a branch whose
//! condition is not statically decidable, or a loop all produce
//! [`ScalarValue::Unknown`] (or a [`ScalarEvalRefusal`]) — never a guess. In
//! particular this evaluator deliberately does **not** seed `exec` with a full
//! wave mask the way SharpEmu's concrete interpreter does: at shader entry
//! `exec` is a hardware-supplied lane mask, so pretending it is all-ones would
//! be a guess.

use crate::hw_regs::UserSgprInfo;
use crate::shader::types::ShaderOperandType as O;
use crate::shader::types::{ShaderInstruction, ShaderInstructionType, ShaderOperand};

use ShaderInstructionType as T;

/// Scalar registers RDNA2 exposes to `operand_parse` as `ShaderOperandType::Sgpr`
/// (encodings 0..=103). `vcc`, `exec`, `m0` and `scc` are *not* in this file —
/// Raeen's IR gives them their own operand types, so [`ScalarState`] tracks them
/// as named fields instead of SharpEmu's flat 256-entry array.
pub const SGPR_COUNT: usize = 104;

/// A single 32-bit lattice element: either a proven dispatch-time constant or
/// bottom. There is deliberately no "maybe" / range / congruence rung — a
/// two-point lattice is all the address folding needs, and it makes every
/// combinator trivially checkable by eye.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ScalarValue {
    /// Not provable at translate time. Absorbing: any operation with an
    /// `Unknown` input produces `Unknown`.
    #[default]
    Unknown,
    /// Proven equal to this value for every execution of the instruction.
    Known(u32),
}

impl ScalarValue {
    /// The proven value, or `None`.
    #[must_use]
    pub const fn known(self) -> Option<u32> {
        match self {
            Self::Known(v) => Some(v),
            Self::Unknown => None,
        }
    }

    #[must_use]
    pub const fn is_known(self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Lift a total unary function. `Unknown` in, `Unknown` out.
    #[must_use]
    pub fn map(self, f: impl FnOnce(u32) -> u32) -> Self {
        match self {
            Self::Known(v) => Self::Known(f(v)),
            Self::Unknown => Self::Unknown,
        }
    }

    /// Lift a total binary function. Either input `Unknown` ⇒ `Unknown`.
    ///
    /// This is the only way this module combines two values, which is what
    /// makes unknown-propagation a structural property rather than something
    /// each opcode arm has to remember.
    #[must_use]
    pub fn zip(self, other: Self, f: impl FnOnce(u32, u32) -> u32) -> Self {
        match (self, other) {
            (Self::Known(a), Self::Known(b)) => Self::Known(f(a, b)),
            _ => Self::Unknown,
        }
    }
}

impl From<Option<u32>> for ScalarValue {
    fn from(value: Option<u32>) -> Self {
        match value {
            Some(v) => Self::Known(v),
            None => Self::Unknown,
        }
    }
}

/// A 64-bit lattice element held as an explicit dword pair, matching how the
/// hardware (and SharpEmu's `WriteScalarPair`) stores 64-bit scalars. Half a
/// pair can be known while the other half is not — `s_mov_b64 s[4:5], s[8:9]`
/// with only `s8` proven keeps `s4` proven — so this is a pair of
/// [`ScalarValue`], not a `ScalarValue<u64>`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScalarValue64 {
    pub lo: ScalarValue,
    pub hi: ScalarValue,
}

impl ScalarValue64 {
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            lo: ScalarValue::Unknown,
            hi: ScalarValue::Unknown,
        }
    }

    #[must_use]
    pub const fn known(value: u64) -> Self {
        Self {
            lo: ScalarValue::Known(value as u32),
            hi: ScalarValue::Known((value >> 32) as u32),
        }
    }

    /// The proven 64-bit value — only when **both** halves are proven.
    #[must_use]
    pub const fn full(self) -> Option<u64> {
        match (self.lo, self.hi) {
            (ScalarValue::Known(lo), ScalarValue::Known(hi)) => Some((hi as u64) << 32 | lo as u64),
            _ => None,
        }
    }

    #[must_use]
    fn map(self, f: impl FnOnce(u64) -> u64) -> Self {
        match self.full() {
            Some(v) => Self::known(f(v)),
            None => Self::unknown(),
        }
    }

    #[must_use]
    fn zip(self, other: Self, f: impl FnOnce(u64, u64) -> u64) -> Self {
        match (self.full(), other.full()) {
            (Some(a), Some(b)) => Self::known(f(a, b)),
            _ => Self::unknown(),
        }
    }
}

/// Why the evaluator could not produce a state for the requested instruction.
///
/// Recorded so a caller's log line names the exact wall rather than an
/// undifferentiated "unknown"; the caller keeps its own refusal message.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarEvalRefusal {
    /// A conditional branch whose condition is not a proven constant. Both
    /// successors are live, so no single register file describes the target.
    UndecidableBranch { pc: u32 },
    /// A branch that can re-enter the walked prefix, i.e. the target
    /// instruction may execute more than once with different scalar state. A
    /// per-PC snapshot cannot represent that.
    Loop { pc: u32, target_pc: u32 },
    /// `s_setpc_b64` / `s_swappc_b64`: the next PC is a runtime value.
    IndirectBranch { pc: u32 },
    /// `s_endpgm` (or the end of the decoded stream) came first — the target
    /// instruction is not reachable along the proven path.
    Unreachable,
    /// The walk exceeded [`STEP_BUDGET`]. Defensive only: every backward branch
    /// is already refused as [`Self::Loop`].
    StepBudget,
    /// The requested instruction index is outside the instruction slice.
    BadIndex,
}

/// Maximum instructions the deterministic walk will execute. Real Gen5 shaders
/// are far below this; the budget exists so a malformed decode cannot spin.
pub const STEP_BUDGET: usize = 1 << 16;

/// The scalar register file as a lattice, plus the wave state the folded
/// opcodes read and write.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScalarState {
    sgpr: [ScalarValue; SGPR_COUNT],
    vcc: ScalarValue64,
    exec: ScalarValue64,
    m0: ScalarValue,
    /// SharpEmu's `scalarConditionCode`. `None` = not proven.
    scc: Option<bool>,
}

impl Default for ScalarState {
    fn default() -> Self {
        Self {
            sgpr: [ScalarValue::Unknown; SGPR_COUNT],
            vcc: ScalarValue64::unknown(),
            exec: ScalarValue64::unknown(),
            m0: ScalarValue::Unknown,
            scc: None,
        }
    }
}

impl ScalarState {
    /// Seed the live-in user-data registers from the PM4-latched values.
    ///
    /// `shift` is the NGG scalar rebase (`rebase_ngg_constant_sharps`'s
    /// `NGG_SCALAR_BASE`, 8 for a gs-prolog vertex stage, 0 elsewhere): shader
    /// register `N` is hardware user-data slot `N - shift`. Registers below
    /// `shift`, and every register at or beyond `count`, stay `Unknown` — the
    /// former are hardware-supplied prolog values and the latter are the
    /// system registers (tgid/tid/wave id) the hardware writes per dispatch.
    #[must_use]
    pub fn with_user_data(user_sgpr: Option<&UserSgprInfo>, shift: i32) -> Self {
        let mut state = Self::default();
        let Some(user_sgpr) = user_sgpr else {
            return state;
        };
        let count = (user_sgpr.count as usize).min(UserSgprInfo::SGPRS_MAX);
        for slot in 0..count {
            let Ok(reg) = usize::try_from(slot as i64 + i64::from(shift)) else {
                continue;
            };
            if reg < SGPR_COUNT {
                state.sgpr[reg] = ScalarValue::Known(user_sgpr.value[slot]);
            }
        }
        state
    }

    /// Read scalar register `reg`. Out-of-range ⇒ `Unknown` (never a panic:
    /// `register_id` comes from a guest-controlled instruction word).
    #[must_use]
    pub fn sgpr(&self, reg: i32) -> ScalarValue {
        usize::try_from(reg)
            .ok()
            .and_then(|r| self.sgpr.get(r).copied())
            .unwrap_or(ScalarValue::Unknown)
    }

    /// The proven condition code, or `None`.
    #[must_use]
    pub const fn scc(&self) -> Option<bool> {
        self.scc
    }

    fn set_sgpr(&mut self, reg: i32, value: ScalarValue) {
        if let Ok(r) = usize::try_from(reg) {
            if let Some(slot) = self.sgpr.get_mut(r) {
                *slot = value;
            }
        }
    }

    /// Read a 32-bit source operand.
    ///
    /// `Vgpr` is lane-varying, `Unknown` is an undecoded operand: both are
    /// bottom. `Null` reads as zero (RDNA2 `null` source), matching SharpEmu's
    /// encoded-constant 125 arm.
    #[must_use]
    fn read32(&self, op: &ShaderOperand) -> ScalarValue {
        match op.type_ {
            O::LiteralConstant | O::IntegerInlineConstant | O::FloatInlineConstant => {
                ScalarValue::Known(op.constant.u)
            }
            O::Null => ScalarValue::Known(0),
            O::Sgpr => self.sgpr(op.register_id),
            O::VccLo => self.vcc.lo,
            O::VccHi => self.vcc.hi,
            O::ExecLo => self.exec.lo,
            O::ExecHi => self.exec.hi,
            O::M0 => self.m0,
            // SharpEmu encoded constants 251/252/253 (VCCZ/EXECZ/SCC). Raeen's
            // parser only materialises EXECZ and SCC as operand types.
            O::ExecZ => self
                .exec
                .full()
                .map(|e| u32::from(e == 0))
                .map_or(ScalarValue::Unknown, ScalarValue::Known),
            O::Scc => self
                .scc
                .map(u32::from)
                .map_or(ScalarValue::Unknown, ScalarValue::Known),
            O::Vgpr | O::Unknown => ScalarValue::Unknown,
        }
    }

    /// Read a 64-bit source operand (SharpEmu `TryEvaluateScalarOperand64`).
    ///
    /// A 32-bit *inline* constant in a 64-bit slot sign-extends (SharpEmu's
    /// `encoded is >= 193 and <= 208` arm — the negative inline range); a
    /// 32-bit *literal* zero-extends. Raeen's `operand_parse` has already
    /// sign-extended the negative inline range into `constant.u`, so the test
    /// here is on the sign bit of that decoded value.
    #[must_use]
    fn read64(&self, op: &ShaderOperand) -> ScalarValue64 {
        match op.type_ {
            O::IntegerInlineConstant => ScalarValue64 {
                lo: ScalarValue::Known(op.constant.u),
                hi: ScalarValue::Known(if op.constant.i() < 0 { u32::MAX } else { 0 }),
            },
            O::LiteralConstant | O::FloatInlineConstant => ScalarValue64 {
                lo: ScalarValue::Known(op.constant.u),
                hi: ScalarValue::Known(0),
            },
            O::Null => ScalarValue64::known(0),
            O::Sgpr => ScalarValue64 {
                lo: self.sgpr(op.register_id),
                hi: self.sgpr(op.register_id.wrapping_add(1)),
            },
            O::VccLo => self.vcc,
            O::ExecLo => self.exec,
            _ => ScalarValue64::unknown(),
        }
    }

    /// Write a 32-bit destination operand.
    fn write32(&mut self, op: &ShaderOperand, value: ScalarValue) {
        match op.type_ {
            O::Sgpr => self.set_sgpr(op.register_id, value),
            O::VccLo => self.vcc.lo = value,
            O::VccHi => self.vcc.hi = value,
            O::ExecLo => self.exec.lo = value,
            O::ExecHi => self.exec.hi = value,
            O::M0 => self.m0 = value,
            // A write we cannot model must not leave a stale proven value
            // anywhere; nothing else is addressable from a 32-bit sdst.
            _ => {}
        }
    }

    /// Write a 64-bit destination operand (SharpEmu `WriteScalarPair`).
    fn write64(&mut self, op: &ShaderOperand, value: ScalarValue64) {
        match op.type_ {
            O::Sgpr => {
                self.set_sgpr(op.register_id, value.lo);
                self.set_sgpr(op.register_id.wrapping_add(1), value.hi);
            }
            O::VccLo => self.vcc = value,
            O::ExecLo => self.exec = value,
            _ => {}
        }
    }

    /// Invalidate everything `inst` may write. Called for every instruction the
    /// folder does not model, so an unmodelled definition can never be read as
    /// its stale pre-definition value.
    fn kill_destinations(&mut self, inst: &ShaderInstruction) {
        for dst in [&inst.dst, &inst.dst2] {
            let span = dst.size.max(1);
            match dst.type_ {
                O::Sgpr => {
                    for i in 0..span {
                        self.set_sgpr(dst.register_id.wrapping_add(i), ScalarValue::Unknown);
                    }
                }
                O::VccLo => {
                    self.vcc.lo = ScalarValue::Unknown;
                    if span > 1 {
                        self.vcc.hi = ScalarValue::Unknown;
                    }
                }
                O::VccHi => self.vcc.hi = ScalarValue::Unknown,
                O::ExecLo => {
                    self.exec.lo = ScalarValue::Unknown;
                    if span > 1 {
                        self.exec.hi = ScalarValue::Unknown;
                    }
                }
                O::ExecHi => self.exec.hi = ScalarValue::Unknown,
                O::M0 => self.m0 = ScalarValue::Unknown,
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// per-opcode folding
// ---------------------------------------------------------------------------

/// SharpEmu `SignedAddOverflow` (L1736).
const fn signed_add_overflow(left: u32, right: u32, result: u32) -> bool {
    ((left ^ result) & (right ^ result) & 0x8000_0000) != 0
}

/// SharpEmu `SignedSubOverflow` (L1739).
const fn signed_sub_overflow(left: u32, right: u32, result: u32) -> bool {
    ((left ^ right) & (left ^ result) & 0x8000_0000) != 0
}

/// SharpEmu `ReverseBits` (L2347).
const fn reverse_bits(value: u32) -> u32 {
    value.reverse_bits()
}

/// `s_bfe_u32` / `s_bfe_i32` field extraction: `right[4:0]` is the offset and
/// `right[22:16]` the width, clamped so the field stays inside the dword
/// (SharpEmu L1503-1520).
const fn bfe32(left: u32, right: u32, signed: bool) -> u32 {
    let offset = (right & 31) as i32;
    let raw_width = ((right >> 16) & 0x7f) as i32;
    let width = if raw_width < 32 - offset {
        raw_width
    } else {
        32 - offset
    };
    if width <= 0 {
        return 0;
    }
    if signed {
        // Shift the field to the top, then arithmetic-shift it back down.
        (((left << (32 - width - offset)) as i32) >> (32 - width)) as u32
    } else {
        (left >> offset) & (u32::MAX >> (32 - width))
    }
}

/// 64-bit `s_bfe_u64` sibling (SharpEmu L1226-1267). `control[5:0]` = offset,
/// `control[22:16]` = width.
const fn bfe64(source: u64, control: u32, signed: bool) -> u64 {
    let offset = (control & 63) as i32;
    let raw_width = ((control >> 16) & 0x7f) as i32;
    let width = if raw_width < 64 - offset {
        raw_width
    } else {
        64 - offset
    };
    if width <= 0 {
        return 0;
    }
    let mut value = source >> offset;
    if width < 64 {
        value &= u64::MAX >> (64 - width);
        if signed {
            value = (((value << (64 - width)) as i64) >> (64 - width)) as u64;
        }
    }
    value
}

/// A 32-bit ALU result plus the condition code it publishes. `scc: None` means
/// "this opcode does not touch SCC"; `Some(None)` means "it does, but the value
/// is not provable".
struct Folded32 {
    value: ScalarValue,
    scc: Option<Option<bool>>,
}

impl Folded32 {
    const fn plain(value: ScalarValue) -> Self {
        Self { value, scc: None }
    }

    /// SCC = (result != 0) — the `s_and`/`s_or`/`s_lshl`/… convention.
    const fn nz(value: ScalarValue) -> Self {
        let scc = match value {
            ScalarValue::Known(v) => Some(Some(v != 0)),
            ScalarValue::Unknown => Some(None),
        };
        Self { value, scc }
    }

    /// SCC computed from the inputs (carry / borrow / overflow / compare).
    fn with(value: ScalarValue, scc: Option<bool>) -> Self {
        Self {
            value,
            scc: Some(scc),
        }
    }
}

/// Fold a two-source 32-bit scalar ALU opcode. `None` = not modelled here.
///
/// Mirrors SharpEmu `TryExecuteScalarAlu`'s `switch` (L1394-1571) op for op,
/// including which arms publish SCC. Every arm goes through
/// [`ScalarValue::zip`], so an unknown operand cannot be silently defaulted.
fn fold_sop2(
    type_: ShaderInstructionType,
    left: ScalarValue,
    right: ScalarValue,
    scc_in: Option<bool>,
) -> Option<Folded32> {
    use ScalarValue as V;

    /// `s_lshlN_add_u32`: the low 32 bits of `(left << n) + right`, plus the
    /// 33-bit carry-out that lands in SCC. SharpEmu
    /// `Gen5ShaderScalarEvaluator.cs` L1525-1552 uses this one body for all
    /// four shifts.
    fn lshl_add(a: u32, b: u32, n: u32) -> (u32, bool) {
        let wide = (u64::from(a) << n) + u64::from(b);
        (wide as u32, wide > u64::from(u32::MAX))
    }

    // Carry/borrow/overflow flags need both operands, so they are derived from
    // the same `zip` guard as the value.
    let both = |f: fn(u32, u32) -> (u32, bool)| -> Folded32 {
        match (left, right) {
            (V::Known(a), V::Known(b)) => {
                let (value, scc) = f(a, b);
                Folded32::with(V::Known(value), Some(scc))
            }
            _ => Folded32::with(V::Unknown, None),
        }
    };

    Some(match type_ {
        T::SAddU32 => both(|a, b| {
            let wide = u64::from(a) + u64::from(b);
            (wide as u32, wide > u64::from(u32::MAX))
        }),
        T::SSubU32 => both(|a, b| (a.wrapping_sub(b), b > a)),
        T::SAddI32 => both(|a, b| {
            let r = (a as i32).wrapping_add(b as i32) as u32;
            (r, signed_add_overflow(a, b, r))
        }),
        T::SSubI32 => both(|a, b| {
            let r = (a as i32).wrapping_sub(b as i32) as u32;
            (r, signed_sub_overflow(a, b, r))
        }),
        // Carry-in comes from SCC, so an unproven SCC makes the result
        // unknown even with both operands proven.
        T::SAddcU32 => match (left, right, scc_in) {
            (V::Known(a), V::Known(b), Some(carry)) => {
                let wide = u64::from(a) + u64::from(b) + u64::from(carry);
                Folded32::with(V::Known(wide as u32), Some(wide > u64::from(u32::MAX)))
            }
            _ => Folded32::with(V::Unknown, None),
        },
        // SharpEmu's `SCselectB32` does not publish SCC (it consumes it).
        T::SCselectB32 => Folded32::plain(match scc_in {
            Some(true) => left,
            Some(false) => right,
            // Both arms proven and equal ⇒ the choice does not matter. This is
            // the only place the evaluator resolves a value without resolving
            // its condition, and it is exact, not a guess.
            None => match (left, right) {
                (V::Known(a), V::Known(b)) if a == b => V::Known(a),
                _ => V::Unknown,
            },
        }),
        T::SAndB32 => Folded32::nz(left.zip(right, |a, b| a & b)),
        T::SOrB32 => Folded32::nz(left.zip(right, |a, b| a | b)),
        T::SLshlB32 => Folded32::nz(left.zip(right, |a, b| a << (b & 31))),
        T::SLshrB32 => Folded32::nz(left.zip(right, |a, b| a >> (b & 31))),
        T::SBfeU32 => Folded32::nz(left.zip(right, |a, b| bfe32(a, b, false))),
        // SharpEmu: `SBfmB32` and `SMulI32`/`SMulHiU32`/`SPackLl*` leave SCC.
        T::SBfmB32 => Folded32::plain(left.zip(right, |a, b| {
            let width = a & 31;
            let offset = b & 31;
            if width == 0 {
                0
            } else {
                ((1u32 << width) - 1) << offset
            }
        })),
        T::SMulI32 => {
            Folded32::plain(left.zip(right, |a, b| (a as i32).wrapping_mul(b as i32) as u32))
        }
        T::SMulHiU32 => {
            Folded32::plain(left.zip(right, |a, b| ((u64::from(a) * u64::from(b)) >> 32) as u32))
        }
        // SharpEmu `Gen5ShaderScalarEvaluator.cs` L1525-1552 folds all four
        // shifts through one body: `wide = (left << N) + right`, result is the
        // low 32 bits and SCC the 33-bit carry-out.
        T::SLshl1AddU32 => both(|a, b| lshl_add(a, b, 1)),
        T::SLshl2AddU32 => both(|a, b| lshl_add(a, b, 2)),
        T::SLshl3AddU32 => both(|a, b| lshl_add(a, b, 3)),
        T::SLshl4AddU32 => both(|a, b| lshl_add(a, b, 4)),
        T::SPackLlB32B16 => Folded32::plain(left.zip(right, |a, b| (a & 0xffff) | (b << 16))),
        _ => return None,
    })
}

/// Fold a 64-bit bitwise / shift / field opcode. `None` = not modelled here.
/// Mirrors SharpEmu L1170-1332.
fn fold_sop2_64(
    type_: ShaderInstructionType,
    left: ScalarValue64,
    right: ScalarValue64,
    shift_amount: ScalarValue,
    scc_in: Option<bool>,
) -> Option<ScalarValue64> {
    Some(match type_ {
        T::SMovB64 => left,
        T::SNotB64 => left.map(|v| !v),
        T::SWqmB64 => left.map(|v| {
            // Quad-any expansion: OR the four lanes of each quad, then splat.
            let quad_any = (v | (v >> 1) | (v >> 2) | (v >> 3)) & 0x1111_1111_1111_1111;
            quad_any.wrapping_mul(0xf)
        }),
        T::SAndB64 => left.zip(right, |a, b| a & b),
        T::SOrB64 => left.zip(right, |a, b| a | b),
        T::SXorB64 => left.zip(right, |a, b| a ^ b),
        T::SAndn2B64 => left.zip(right, |a, b| a & !b),
        T::SOrn2B64 => left.zip(right, |a, b| a | !b),
        T::SNandB64 => left.zip(right, |a, b| !(a & b)),
        T::SNorB64 => left.zip(right, |a, b| !(a | b)),
        T::SXnorB64 => left.zip(right, |a, b| !(a ^ b)),
        T::SCselectB64 => match scc_in {
            Some(true) => left,
            Some(false) => right,
            None => match (left.full(), right.full()) {
                (Some(a), Some(b)) if a == b => ScalarValue64::known(a),
                _ => ScalarValue64::unknown(),
            },
        },
        T::SLshlB64 => match (left.full(), shift_amount.known()) {
            (Some(v), Some(s)) => ScalarValue64::known(v << (s & 63)),
            _ => ScalarValue64::unknown(),
        },
        T::SLshrB64 => match (left.full(), shift_amount.known()) {
            (Some(v), Some(s)) => ScalarValue64::known(v >> (s & 63)),
            _ => ScalarValue64::unknown(),
        },
        T::SBfeU64 => match (left.full(), shift_amount.known()) {
            (Some(v), Some(c)) => ScalarValue64::known(bfe64(v, c, false)),
            _ => ScalarValue64::unknown(),
        },
        _ => return None,
    })
}

/// Fold a SOPC scalar compare into SCC (SharpEmu `TryExecuteScalarCompare`
/// L1780-1795). `None` = not a compare.
fn fold_compare(
    type_: ShaderInstructionType,
    left: ScalarValue,
    right: ScalarValue,
) -> Option<Option<bool>> {
    let cmp: fn(u32, u32) -> bool = match type_ {
        T::SCmpEqI32 => |a, b| a as i32 == b as i32,
        T::SCmpLgI32 => |a, b| a as i32 != b as i32,
        T::SCmpGtI32 => |a, b| (a as i32) > (b as i32),
        T::SCmpGeI32 => |a, b| (a as i32) >= (b as i32),
        T::SCmpLtI32 => |a, b| (a as i32) < (b as i32),
        T::SCmpLeI32 => |a, b| (a as i32) <= (b as i32),
        T::SCmpEqU32 => |a, b| a == b,
        T::SCmpLgU32 => |a, b| a != b,
        T::SCmpGtU32 => |a, b| a > b,
        T::SCmpGeU32 => |a, b| a >= b,
        T::SCmpLtU32 => |a, b| a < b,
        T::SCmpLeU32 => |a, b| a <= b,
        _ => return None,
    };
    Some(match (left, right) {
        (ScalarValue::Known(a), ScalarValue::Known(b)) => Some(cmp(a, b)),
        _ => None,
    })
}

/// The save-exec family: `sdst = exec; exec = <op>(ssrc0, exec)`
/// (SharpEmu `TryExecuteSaveExecScalarAlu`). Only the three forms Raeen's
/// parser decodes are here. Since `exec` starts `Unknown`, these normally
/// yield `Unknown` — they are modelled anyway so a shader that first pins
/// `exec` to a constant keeps folding.
fn fold_saveexec(
    type_: ShaderInstructionType,
    source: ScalarValue64,
    exec: ScalarValue64,
) -> Option<ScalarValue64> {
    Some(match type_ {
        T::SAndSaveexecB64 => exec.zip(source, |e, s| e & s),
        // 0x28 `s_orn2_saveexec_b64`: exec = ssrc0 | ~exec.
        T::SOrn2SaveexecB64 => exec.zip(source, |e, s| s | !e),
        // 0x37 `s_andn1_saveexec_b64`: exec = ~ssrc0 & exec.
        T::SAndn1SaveexecB64 => exec.zip(source, |e, s| !s & e),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// the walk
// ---------------------------------------------------------------------------

/// Is `type_` a scalar branch, and if so what is its target PC?
///
/// Target = `pc + 4 + simm * 4`, already pre-multiplied into `src[0]` by
/// `shader_parse_sopp` (same arithmetic as `ShaderLabel::from_instruction`).
fn branch_target(inst: &ShaderInstruction) -> Option<u32> {
    matches!(
        inst.type_,
        T::SBranch
            | T::SCbranchScc0
            | T::SCbranchScc1
            | T::SCbranchVccz
            | T::SCbranchVccnz
            | T::SCbranchExecz
    )
    .then(|| {
        inst.pc
            .wrapping_add(4)
            .wrapping_add(inst.src[0].constant.i() as u32)
    })
}

/// Run the scalar stream from entry and return the register file **as it is
/// immediately before** `instructions[target]` executes.
///
/// # Soundness contract
///
/// The returned state describes *every* execution of `instructions[target]`,
/// which is what a per-PC snapshot needs. Two gates enforce that:
///
/// 1. **Determinism.** The walk only follows a branch whose condition is a
///    proven constant, so exactly one execution trace leaves entry. An
///    undecidable condition, an indirect PC write, or `s_endpgm` before the
///    target is a refusal, not a guess. Because the trace is unique, the target
///    cannot be entered with some *other* register file — only with this one.
/// 2. **No cycle through the target.** A unique trace can still visit the
///    target more than once, which a per-PC snapshot cannot represent. Two
///    checks cover that: the walk refuses any branch it would *take* backwards,
///    and a pre-pass refuses any branch sitting **after** the target whose
///    destination is at or before it. The second one has to look past the
///    target because the re-entering edge of a descriptor loop is exactly there
///    (`load; s_add; s_cbranch_scc1 back`) and the walk stops before reaching
///    it.
///
/// Within those gates every unmodelled definition is killed
/// ([`ScalarState::kill_destinations`]), so a value that depends on guest
/// memory (`s_load_*`, `s_buffer_load_*`, `global_load_*`), on a lane
/// (`v_readfirstlane_b32`, a VOPC writing an SGPR), or on any opcode this
/// module does not fold reads back as [`ScalarValue::Unknown`].
pub fn evaluate_before(
    instructions: &[ShaderInstruction],
    target: usize,
    user_sgpr: Option<&UserSgprInfo>,
    shift: i32,
) -> Result<ScalarState, ScalarEvalRefusal> {
    let Some(target_inst) = instructions.get(target) else {
        return Err(ScalarEvalRefusal::BadIndex);
    };
    let target_pc = target_inst.pc;

    // Gate 2a: an edge that sits after the target and jumps back to or before
    // it can run the target a second time. The walk stops before that edge, so
    // it has to be found by inspection.
    for inst in instructions {
        if inst.pc <= target_pc {
            continue;
        }
        if let Some(dst) = branch_target(inst) {
            if dst <= target_pc {
                return Err(ScalarEvalRefusal::Loop {
                    pc: inst.pc,
                    target_pc: dst,
                });
            }
        }
    }

    let mut state = ScalarState::with_user_data(user_sgpr, shift);
    let mut index = 0usize;
    for _ in 0..STEP_BUDGET {
        if index == target {
            return Ok(state);
        }
        let Some(inst) = instructions.get(index) else {
            return Err(ScalarEvalRefusal::Unreachable);
        };

        match inst.type_ {
            T::SEndpgm => return Err(ScalarEvalRefusal::Unreachable),
            T::SSetpcB64 | T::SSwappcB64 => {
                return Err(ScalarEvalRefusal::IndirectBranch { pc: inst.pc });
            }
            _ => {}
        }

        if let Some(dst_pc) = branch_target(inst) {
            let taken = match inst.type_ {
                T::SBranch => Some(true),
                T::SCbranchScc0 => state.scc.map(|scc| !scc),
                T::SCbranchScc1 => state.scc,
                T::SCbranchVccz => state.vcc.full().map(|v| v == 0),
                T::SCbranchVccnz => state.vcc.full().map(|v| v != 0),
                T::SCbranchExecz => state.exec.full().map(|v| v == 0),
                _ => None,
            };
            let Some(taken) = taken else {
                return Err(ScalarEvalRefusal::UndecidableBranch { pc: inst.pc });
            };
            if taken {
                // Gate 2b: a taken backward edge is a cycle. Even with a proven
                // condition the trace would revisit instructions (and the
                // condition itself might flip on the next trip), so no single
                // register file describes what follows.
                if dst_pc <= inst.pc {
                    return Err(ScalarEvalRefusal::Loop {
                        pc: inst.pc,
                        target_pc: dst_pc,
                    });
                }
                let Some(next) = instructions.iter().position(|i| i.pc == dst_pc) else {
                    return Err(ScalarEvalRefusal::Unreachable);
                };
                index = next;
            } else {
                index += 1;
            }
            continue;
        }

        step(&mut state, inst);
        index += 1;
    }
    Err(ScalarEvalRefusal::StepBudget)
}

/// Execute one non-branch instruction against the lattice.
fn step(state: &mut ScalarState, inst: &ShaderInstruction) {
    // Pure no-ops: nothing to fold and nothing to kill.
    if matches!(
        inst.type_,
        T::SNop
            | T::SWaitcnt
            | T::SBarrier
            | T::SVersion
            | T::SInstPrefetch
            | T::SSendmsg
            | T::SBranch
    ) {
        return;
    }

    // SOPC compares write only SCC.
    if let Some(scc) = fold_compare(
        inst.type_,
        state.read32(&inst.src[0]),
        state.read32(&inst.src[1]),
    ) {
        state.scc = scc;
        return;
    }

    // `s_getpc_b64` — the parser already resolved the following instruction's
    // absolute guest address into src[0]/src[1] as literals, so this is an
    // ordinary 64-bit constant move.
    if inst.type_ == T::SGetpcB64 {
        state.write64(
            &inst.dst,
            ScalarValue64 {
                lo: state.read32(&inst.src[0]),
                hi: state.read32(&inst.src[1]),
            },
        );
        return;
    }

    // Save-exec: sdst takes the OLD exec, exec takes the combined mask.
    if let Some(new_exec) = fold_saveexec(inst.type_, state.read64(&inst.src[0]), state.exec) {
        let old_exec = state.exec;
        state.write64(&inst.dst, old_exec);
        state.exec = new_exec;
        state.scc = new_exec.full().map(|e| e != 0);
        return;
    }

    // 64-bit bitwise / shift / select / field ops.
    if let Some(value) = fold_sop2_64(
        inst.type_,
        state.read64(&inst.src[0]),
        state.read64(&inst.src[1]),
        state.read32(&inst.src[1]),
        state.scc,
    ) {
        state.write64(&inst.dst, value);
        // SharpEmu publishes SCC for every 64-bit op it folds except the plain
        // `s_mov_b64` (and `s_cselect_b64`, which consumes it).
        if !matches!(inst.type_, T::SMovB64 | T::SCselectB64) {
            state.scc = value.full().map(|v| v != 0);
        }
        return;
    }

    // 32-bit moves and the SOPK immediates.
    match inst.type_ {
        // `s_movk_i32`/`s_mulk_i32` immediates are already sign-extended into
        // `constant.u` by `shader_parse_sopk`.
        T::SMovB32 | T::SMovkI32 => {
            state.write32(&inst.dst, state.read32(&inst.src[0]));
            return;
        }
        T::SMulkI32 => {
            // Read-modify-write on sdst.
            let prior = state.read32(&inst.dst);
            let value = prior.zip(state.read32(&inst.src[0]), |a, b| {
                (a as i32).wrapping_mul(b as i32) as u32
            });
            state.write32(&inst.dst, value);
            return;
        }
        T::SBrevB32 => {
            // SharpEmu folds this in the one-source group; the parser marks it
            // as not writing SCC.
            let value = state.read32(&inst.src[0]).map(reverse_bits);
            state.write32(&inst.dst, value);
            return;
        }
        T::SBitset1B32 => {
            // Read-modify-write: the source selects a destination bit. SCC is
            // explicitly preserved.
            let value = state
                .read32(&inst.dst)
                .zip(state.read32(&inst.src[0]), |dst, bit| {
                    dst | (1u32 << (bit & 31))
                });
            state.write32(&inst.dst, value);
            return;
        }
        _ => {}
    }

    // Two-source 32-bit ALU.
    if let Some(folded) = fold_sop2(
        inst.type_,
        state.read32(&inst.src[0]),
        state.read32(&inst.src[1]),
        state.scc,
    ) {
        state.write32(&inst.dst, folded.value);
        if let Some(scc) = folded.scc {
            state.scc = scc;
        }
        return;
    }

    // Everything else: a scalar memory load, a vector op with an SGPR
    // destination, a lane read, an opcode this module does not fold. Its
    // definitions become unknown — the whole point of the lattice.
    state.kill_destinations(inst);

    // An unmodelled instruction may also set SCC (e.g. a scalar op added to
    // the parser later without a fold arm here). Conservatively drop it if the
    // instruction writes any scalar destination at all.
    if !matches!(inst.dst.type_, O::Unknown) || !matches!(inst.dst2.type_, O::Unknown) {
        state.scc = None;
    }
}

/// Prove the dispatch-time value of scalar register `reg` as read by
/// `instructions[at]`.
///
/// Two independent routes, tried in order:
///
/// 1. **Never written.** If no instruction in the program defines `reg`, its
///    value is still the live-in the PM4 stream latched, whatever the control
///    flow does. This is the SRT / global-table pointer ABI and is sound
///    without any CFG reasoning at all — which is why it is checked first, and
///    why it is strictly stronger than a prefix-only "no earlier writer" test.
/// 2. **Deterministic walk.** Otherwise fold the stream with
///    [`evaluate_before`] and read `reg` out of the resulting lattice.
///
/// `Err` carries why the walk gave up; route 1's failure is reported as
/// `Unknown` inside `Ok` (the walk ran, the register just is not provable).
pub fn resolve_sgpr_before(
    instructions: &[ShaderInstruction],
    at: usize,
    reg: i32,
    user_sgpr: Option<&UserSgprInfo>,
    shift: i32,
) -> Result<ScalarValue, ScalarEvalRefusal> {
    if !instructions.iter().any(|inst| writes_sgpr(inst, reg)) {
        return Ok(live_in_sgpr(reg, user_sgpr, shift));
    }
    Ok(evaluate_before(instructions, at, user_sgpr, shift)?.sgpr(reg))
}

/// Prove a 32-bit scalar operand as read by `instructions[at]`.
///
/// Unlike [`resolve_sgpr_before`], this also serves the named scalar registers
/// (`vcc`, `exec`, `m0`, and `scc`) that RDNA2 permits in an SMEM `soffset`
/// field. They are never treated as live-in constants: the deterministic walk
/// must prove their value through an instruction this evaluator models.
pub fn resolve_scalar_operand_before(
    instructions: &[ShaderInstruction],
    at: usize,
    operand: &ShaderOperand,
    user_sgpr: Option<&UserSgprInfo>,
    shift: i32,
) -> Result<ScalarValue, ScalarEvalRefusal> {
    if operand.type_ == O::Sgpr {
        return resolve_sgpr_before(instructions, at, operand.register_id, user_sgpr, shift);
    }
    Ok(evaluate_before(instructions, at, user_sgpr, shift)?.read32(operand))
}

/// The PM4-latched live-in value of shader scalar register `reg`, or `Unknown`
/// when `reg` is not a captured user-data slot.
#[must_use]
pub fn live_in_sgpr(reg: i32, user_sgpr: Option<&UserSgprInfo>, shift: i32) -> ScalarValue {
    let Some(user_sgpr) = user_sgpr else {
        return ScalarValue::Unknown;
    };
    let Ok(slot) = usize::try_from(reg - shift) else {
        return ScalarValue::Unknown;
    };
    if slot >= UserSgprInfo::SGPRS_MAX || slot >= user_sgpr.count as usize {
        return ScalarValue::Unknown;
    }
    ScalarValue::Known(user_sgpr.value[slot])
}

/// Does `inst` define scalar register `reg` through either destination?
///
/// Mirrors `analysis::writes_sgpr`; duplicated here so the evaluator's
/// never-written proof does not depend on a private helper in another module.
#[must_use]
pub fn writes_sgpr(inst: &ShaderInstruction, reg: i32) -> bool {
    let covers = |op: &ShaderOperand| {
        op.type_ == O::Sgpr && reg >= op.register_id && reg < op.register_id + op.size.max(1)
    };
    covers(&inst.dst) || covers(&inst.dst2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw_regs::UserSgprType;
    use crate::shader::types::{ShaderConstant, shader_instruction_format::Format as F};

    fn sgpr(reg: i32, size: i32) -> ShaderOperand {
        ShaderOperand {
            type_: O::Sgpr,
            register_id: reg,
            size,
            ..Default::default()
        }
    }

    fn named(type_: O) -> ShaderOperand {
        ShaderOperand {
            type_,
            size: 1,
            ..Default::default()
        }
    }

    fn imm(value: u32) -> ShaderOperand {
        ShaderOperand {
            type_: O::IntegerInlineConstant,
            constant: ShaderConstant::from_u(value),
            size: 0,
            ..Default::default()
        }
    }

    fn lit(value: u32) -> ShaderOperand {
        ShaderOperand {
            type_: O::LiteralConstant,
            constant: ShaderConstant::from_u(value),
            size: 0,
            ..Default::default()
        }
    }

    /// `sdst = op(src0, src1)` at `pc`, both 32-bit.
    fn alu2(
        pc: u32,
        type_: ShaderInstructionType,
        dst: i32,
        a: ShaderOperand,
        b: ShaderOperand,
    ) -> ShaderInstruction {
        let mut inst = ShaderInstruction {
            pc,
            type_,
            format: F::SVdstSVsrc0SVsrc1,
            ..Default::default()
        };
        inst.dst = sgpr(dst, 1);
        inst.src[0] = a;
        inst.src[1] = b;
        inst.src_num = 2;
        inst
    }

    fn endpgm(pc: u32) -> ShaderInstruction {
        ShaderInstruction {
            pc,
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        }
    }

    /// A one-dword scalar load: its destination depends on guest memory, so the
    /// evaluator must treat that destination as unknown.
    fn sload(pc: u32, dst: i32, base: i32) -> ShaderInstruction {
        let mut inst = ShaderInstruction {
            pc,
            type_: T::SLoadDword,
            format: F::SdstSbaseSoffsetOffset,
            ..Default::default()
        };
        inst.dst = sgpr(dst, 1);
        inst.src[0] = sgpr(base, 2);
        inst.src[1] = imm(0);
        inst.src[2] = imm(0);
        inst.src_num = 3;
        inst
    }

    fn user_data(pairs: &[(u32, u32)]) -> UserSgprInfo {
        let mut info = UserSgprInfo::default();
        for &(slot, value) in pairs {
            info.set(slot, value, UserSgprType::Unknown);
        }
        info
    }

    /// Run `program` and read `reg` as seen by the LAST instruction.
    fn resolve(program: &[ShaderInstruction], reg: i32, user: &UserSgprInfo) -> ScalarValue {
        let at = program.len() - 1;
        resolve_sgpr_before(program, at, reg, Some(user), 0)
            .expect("walk must not refuse in this fixture")
    }

    // ---- lattice basics ---------------------------------------------------

    #[test]
    fn unknown_is_absorbing_in_both_combinators() {
        let known = ScalarValue::Known(7);
        assert_eq!(
            known.zip(ScalarValue::Unknown, |a, b| a + b),
            ScalarValue::Unknown
        );
        assert_eq!(
            ScalarValue::Unknown.zip(known, |a, b| a + b),
            ScalarValue::Unknown
        );
        assert_eq!(ScalarValue::Unknown.map(|v| v + 1), ScalarValue::Unknown);
        assert_eq!(known.zip(known, |a, b| a + b), ScalarValue::Known(14));
    }

    #[test]
    fn a_64_bit_pair_is_only_full_when_both_halves_are_proven() {
        assert_eq!(
            ScalarValue64::known(0x1122_3344_5566_7788).full(),
            Some(0x1122_3344_5566_7788)
        );
        let half = ScalarValue64 {
            lo: ScalarValue::Known(1),
            hi: ScalarValue::Unknown,
        };
        assert_eq!(half.full(), None);
    }

    #[test]
    fn user_data_seeds_only_captured_slots() {
        let user = user_data(&[(0, 0xaaaa), (1, 0xbbbb)]);
        let state = ScalarState::with_user_data(Some(&user), 0);
        assert_eq!(state.sgpr(0), ScalarValue::Known(0xaaaa));
        assert_eq!(state.sgpr(1), ScalarValue::Known(0xbbbb));
        // Beyond `count` the hardware writes system values per dispatch.
        assert_eq!(state.sgpr(2), ScalarValue::Unknown);
        // Out of range never panics.
        assert_eq!(state.sgpr(-1), ScalarValue::Unknown);
        assert_eq!(state.sgpr(SGPR_COUNT as i32), ScalarValue::Unknown);
    }

    #[test]
    fn the_ngg_shift_moves_user_data_up_and_leaves_the_prolog_registers_unknown() {
        let user = user_data(&[(0, 0x1000), (4, 0x4000)]);
        let state = ScalarState::with_user_data(Some(&user), 8);
        assert_eq!(state.sgpr(8), ScalarValue::Known(0x1000));
        assert_eq!(state.sgpr(12), ScalarValue::Known(0x4000));
        for reg in 0..8 {
            assert_eq!(
                state.sgpr(reg),
                ScalarValue::Unknown,
                "gs-prolog register s{reg} is hardware-supplied, not user data"
            );
        }
    }

    #[test]
    fn exec_is_never_seeded_because_the_entry_lane_mask_is_not_static() {
        let state = ScalarState::with_user_data(Some(&user_data(&[(0, 1)])), 0);
        assert_eq!(state.exec.full(), None);
        assert_eq!(state.vcc.full(), None);
        assert_eq!(state.scc(), None);
    }

    // ---- one test per folded 32-bit opcode: known -> known, unknown -> unknown

    /// `s0` is a captured live-in (known), `s9` is beyond `count` (unknown).
    /// Every folded opcode must produce a proven value from the first and
    /// bottom from the second.
    fn fold_case(type_: ShaderInstructionType, a: u32, b: u32, expect: u32) {
        let user = user_data(&[(0, a), (1, b)]);
        let program = [alu2(0, type_, 4, sgpr(0, 1), sgpr(1, 1)), endpgm(4)];
        assert_eq!(
            resolve(&program, 4, &user),
            ScalarValue::Known(expect),
            "{type_:?} with proven inputs must fold"
        );

        // Same opcode, one operand unproven (s9 >= count).
        let program = [alu2(0, type_, 4, sgpr(0, 1), sgpr(9, 1)), endpgm(4)];
        assert_eq!(
            resolve(&program, 4, &user),
            ScalarValue::Unknown,
            "{type_:?} with an unproven operand must stay unknown"
        );
    }

    #[test]
    fn s_add_u32_folds() {
        fold_case(T::SAddU32, 0x1000, 0x24, 0x1024);
    }

    #[test]
    fn s_sub_u32_folds() {
        fold_case(T::SSubU32, 0x1024, 0x24, 0x1000);
    }

    #[test]
    fn s_add_i32_folds_with_wrapping() {
        fold_case(T::SAddI32, 0xffff_ffff, 2, 1);
    }

    #[test]
    fn s_sub_i32_folds() {
        fold_case(T::SSubI32, 5, 9, (-4i32) as u32);
    }

    #[test]
    fn s_lshl_b32_folds_and_masks_the_shift_to_five_bits() {
        fold_case(T::SLshlB32, 3, 4, 0x30);
        // Shift counts are taken mod 32 (RDNA2), not saturated.
        fold_case(T::SLshlB32, 1, 33, 2);
    }

    #[test]
    fn s_lshr_b32_folds() {
        fold_case(T::SLshrB32, 0x1234_0000, 16, 0x1234);
    }

    #[test]
    fn s_and_b32_folds() {
        fold_case(T::SAndB32, 0xdead_beef, 0x0000_ffff, 0xbeef);
    }

    #[test]
    fn s_or_b32_folds() {
        fold_case(T::SOrB32, 0xdead_0000, 0x0000_beef, 0xdead_beef);
    }

    #[test]
    fn s_mul_i32_folds_signed() {
        fold_case(T::SMulI32, (-3i32) as u32, 7, (-21i32) as u32);
    }

    #[test]
    fn s_mul_hi_u32_folds_the_upper_half() {
        fold_case(T::SMulHiU32, 0x1_0000, 0x1_0000, 1);
    }

    #[test]
    fn s_bfm_b32_builds_a_mask() {
        // width 4, offset 8 -> 0x0000_0f00.
        fold_case(T::SBfmB32, 4, 8, 0x0f00);
        // width 0 is an empty mask, not a full one.
        fold_case(T::SBfmB32, 0, 8, 0);
    }

    #[test]
    fn s_bfe_u32_extracts_the_encoded_field() {
        // offset 8, width 4 -> control 0x0004_0008; source 0xabcd_ef01 -> 0xf.
        fold_case(T::SBfeU32, 0xabcd_ef01, 0x0004_0008, 0xf);
        // A width that runs past the dword clamps rather than shifting by 32.
        fold_case(T::SBfeU32, 0xffff_ffff, 0x0040_0018, 0xff);
        // Zero width is zero, not "the whole register".
        fold_case(T::SBfeU32, 0xffff_ffff, 0x0000_0004, 0);
    }

    #[test]
    fn s_lshl4_add_u32_folds() {
        fold_case(T::SLshl4AddU32, 3, 1, 0x31);
    }

    /// The whole `s_lshlN_add_u32` family folds, so a byte offset computed with
    /// any of the four shifts resolves for the V#/pointer capture passes rather
    /// than leaving them on a named refusal. Semantics from SharpEmu
    /// `Gen5ShaderScalarEvaluator.cs` L1525-1552.
    #[test]
    fn every_s_lshl_n_add_u32_shift_folds() {
        // (left << N) + right, for left = 3, right = 1.
        fold_case(T::SLshl1AddU32, 3, 1, 7);
        fold_case(T::SLshl2AddU32, 3, 1, 0xd);
        fold_case(T::SLshl3AddU32, 3, 1, 0x19);
        fold_case(T::SLshl4AddU32, 3, 1, 0x31);
        // The measured ASTRO.BOT shape: a dword index scaled to a byte offset
        // (index 5, element size 8) plus a base.
        fold_case(T::SLshl3AddU32, 5, 0x100, 0x128);
        // The result is the LOW 32 bits; the overflow goes to SCC, not the dst.
        fold_case(T::SLshl3AddU32, 0x2000_0000, 0, 0);
    }

    #[test]
    fn s_pack_ll_b32_b16_folds() {
        fold_case(T::SPackLlB32B16, 0x1111_2222, 0x3333, 0x3333_2222);
    }

    #[test]
    fn s_addc_u32_needs_a_proven_carry_in() {
        let user = user_data(&[(0, 1), (1, 2)]);
        // No preceding SCC producer: carry-in unproven -> unknown.
        let program = [alu2(0, T::SAddcU32, 4, sgpr(0, 1), sgpr(1, 1)), endpgm(4)];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Unknown);

        // `s_add_u32 s5, 0xffffffff, 1` sets SCC (carry out), so the following
        // s_addc_u32 folds to 1 + 2 + 1.
        let program = [
            alu2(0, T::SAddU32, 5, lit(0xffff_ffff), imm(1)),
            alu2(4, T::SAddcU32, 4, sgpr(0, 1), sgpr(1, 1)),
            endpgm(8),
        ];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(4));
    }

    #[test]
    fn s_cselect_b32_follows_a_proven_scc_and_refuses_an_unproven_one() {
        let user = user_data(&[(0, 0xaa), (1, 0xbb)]);
        // s_cmp_eq_u32 0, 0 -> SCC = true -> select src0.
        let mut cmp = ShaderInstruction {
            pc: 0,
            type_: T::SCmpEqU32,
            format: F::Ssrc0Ssrc1,
            ..Default::default()
        };
        cmp.src[0] = imm(0);
        cmp.src[1] = imm(0);
        cmp.src_num = 2;
        let program = [
            cmp,
            alu2(4, T::SCselectB32, 4, sgpr(0, 1), sgpr(1, 1)),
            endpgm(8),
        ];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(0xaa));

        // Without a compare, SCC is unproven and the two arms differ.
        let program = [
            alu2(0, T::SCselectB32, 4, sgpr(0, 1), sgpr(1, 1)),
            endpgm(4),
        ];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Unknown);
    }

    #[test]
    fn s_cselect_b32_resolves_when_both_arms_are_the_same_proven_value() {
        // Exact, not a guess: whichever way SCC goes the result is 0x40.
        let user = user_data(&[(0, 0x40), (1, 0x40)]);
        let program = [
            alu2(0, T::SCselectB32, 4, sgpr(0, 1), sgpr(1, 1)),
            endpgm(4),
        ];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(0x40));
    }

    #[test]
    fn s_mov_b32_and_s_movk_i32_fold() {
        let user = user_data(&[(0, 0x1234)]);
        let mut mov = ShaderInstruction {
            pc: 0,
            type_: T::SMovB32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        mov.dst = sgpr(4, 1);
        mov.src[0] = sgpr(0, 1);
        mov.src_num = 1;
        let program = [mov, endpgm(4)];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(0x1234));

        let mut movk = ShaderInstruction {
            pc: 0,
            type_: T::SMovkI32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        movk.dst = sgpr(4, 1);
        movk.src[0] = imm((-8i32) as u32);
        movk.src_num = 1;
        let program = [movk, endpgm(4)];
        assert_eq!(
            resolve(&program, 4, &user),
            ScalarValue::Known((-8i32) as u32)
        );
    }

    #[test]
    fn s_brev_b32_folds() {
        let user = user_data(&[(0, 1)]);
        let mut brev = ShaderInstruction {
            pc: 0,
            type_: T::SBrevB32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        brev.dst = sgpr(4, 1);
        brev.src[0] = sgpr(0, 1);
        brev.src_num = 1;
        let program = [brev, endpgm(4)];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(0x8000_0000));
    }

    #[test]
    fn s_bitset1_b32_reads_modifies_and_writes_its_destination() {
        let user = user_data(&[(4, 1)]);
        let mut bitset = ShaderInstruction {
            pc: 0,
            type_: T::SBitset1B32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        bitset.dst = sgpr(4, 1);
        bitset.src[0] = imm(33); // low five bits select bit 1
        bitset.src_num = 1;
        let program = [bitset, endpgm(4)];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(3));
    }

    #[test]
    fn s_mulk_i32_reads_modifies_and_writes_its_destination() {
        // s4 = live-in 6, then s_mulk_i32 s4, 7 -> 42.
        let user = user_data(&[(4, 6)]);
        let mut mulk = ShaderInstruction {
            pc: 0,
            type_: T::SMulkI32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        mulk.dst = sgpr(4, 1);
        mulk.src[0] = imm(7);
        mulk.src_num = 1;
        let program = [mulk, endpgm(4)];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(42));
    }

    // ---- 64-bit pairs -----------------------------------------------------

    fn alu2_64(
        pc: u32,
        type_: ShaderInstructionType,
        dst: i32,
        a: i32,
        b: i32,
    ) -> ShaderInstruction {
        let mut inst = ShaderInstruction {
            pc,
            type_,
            format: F::Sdst2Ssrc02Ssrc12,
            ..Default::default()
        };
        inst.dst = sgpr(dst, 2);
        inst.src[0] = sgpr(a, 2);
        inst.src[1] = sgpr(b, 2);
        inst.src_num = 2;
        inst
    }

    /// `dst = op(s[0:1])` — a one-source 64-bit op, so there is no second
    /// operand to poison; only the source's own unknown half can.
    fn fold_case_64_unary(type_: ShaderInstructionType, a: u64, expect: u64) {
        let user = user_data(&[(0, a as u32), (1, (a >> 32) as u32)]);
        let program = [alu2_64(0, type_, 8, 0, 0), endpgm(4)];
        assert_eq!(
            ScalarValue64 {
                lo: resolve(&program, 8, &user),
                hi: resolve(&program, 9, &user),
            }
            .full(),
            Some(expect),
            "{type_:?} 64-bit fold"
        );

        // s10:s11 is beyond `count`: an unproven SOURCE must poison the result.
        let program = [alu2_64(0, type_, 8, 10, 10), endpgm(4)];
        assert_eq!(
            ScalarValue64 {
                lo: resolve(&program, 8, &user),
                hi: resolve(&program, 9, &user),
            }
            .full(),
            None,
            "{type_:?} 64-bit unknown propagation"
        );
    }

    /// `dst = op(s[0:1], s[2:3])` with all four halves proven.
    fn fold_case_64(type_: ShaderInstructionType, a: u64, b: u64, expect: u64) {
        let user = user_data(&[
            (0, a as u32),
            (1, (a >> 32) as u32),
            (2, b as u32),
            (3, (b >> 32) as u32),
        ]);
        let program = [alu2_64(0, type_, 8, 0, 2), endpgm(4)];
        let lo = resolve(&program, 8, &user);
        let hi = resolve(&program, 9, &user);
        assert_eq!(
            ScalarValue64 { lo, hi }.full(),
            Some(expect),
            "{type_:?} 64-bit fold"
        );

        // s10:s11 is beyond `count` -> unknown, and it must poison the result.
        let program = [alu2_64(0, type_, 8, 0, 10), endpgm(4)];
        assert_eq!(
            ScalarValue64 {
                lo: resolve(&program, 8, &user),
                hi: resolve(&program, 9, &user),
            }
            .full(),
            None,
            "{type_:?} 64-bit unknown propagation"
        );
    }

    #[test]
    fn s_and_or_xor_b64_fold() {
        fold_case_64(
            T::SAndB64,
            0xffff_0000_ffff_0000,
            0x00ff_00ff_00ff_00ff,
            0x00ff_0000_00ff_0000,
        );
        fold_case_64(
            T::SOrB64,
            0xffff_0000_0000_0000,
            0x0000_0000_0000_ffff,
            0xffff_0000_0000_ffff,
        );
        fold_case_64(
            T::SXorB64,
            0xffff_ffff_ffff_ffff,
            0x0f0f_0f0f_0f0f_0f0f,
            0xf0f0_f0f0_f0f0_f0f0,
        );
    }

    #[test]
    fn s_andn2_orn2_nand_nor_xnor_b64_fold() {
        fold_case_64(T::SAndn2B64, 0xff, 0x0f, 0xf0);
        fold_case_64(T::SOrn2B64, 0, 0xffff_ffff_ffff_fff0, 0xf);
        fold_case_64(T::SNandB64, 0xff, 0x0f, !0x0fu64);
        fold_case_64(T::SNorB64, 0xf0, 0x0f, !0xffu64);
        fold_case_64(T::SXnorB64, 0xff, 0x0f, !0xf0u64);
    }

    #[test]
    fn s_mov_b64_and_s_not_b64_fold() {
        fold_case_64_unary(T::SMovB64, 0x1122_3344_5566_7788, 0x1122_3344_5566_7788);
        fold_case_64_unary(T::SNotB64, 0, u64::MAX);
    }

    #[test]
    fn s_wqm_b64_expands_each_quad() {
        // One active lane in quad 0 -> the whole quad becomes active.
        fold_case_64_unary(T::SWqmB64, 0b0001, 0b1111);
        fold_case_64_unary(T::SWqmB64, 0b0100, 0b1111);
    }

    #[test]
    fn s_lshl_lshr_b64_fold_with_a_32_bit_shift_operand() {
        let user = user_data(&[(0, 1), (1, 0), (2, 8), (3, 0)]);
        let mut shl = ShaderInstruction {
            pc: 0,
            type_: T::SLshlB64,
            format: F::Sdst2Ssrc02Ssrc1,
            ..Default::default()
        };
        shl.dst = sgpr(8, 2);
        shl.src[0] = sgpr(0, 2);
        shl.src[1] = sgpr(2, 1);
        shl.src_num = 2;
        let program = [shl, endpgm(4)];
        assert_eq!(
            ScalarValue64 {
                lo: resolve(&program, 8, &user),
                hi: resolve(&program, 9, &user),
            }
            .full(),
            Some(0x100)
        );

        let mut shr = ShaderInstruction {
            pc: 0,
            type_: T::SLshrB64,
            format: F::Sdst2Ssrc02Ssrc1,
            ..Default::default()
        };
        shr.dst = sgpr(8, 2);
        shr.src[0] = sgpr(0, 2);
        shr.src[1] = sgpr(2, 1);
        shr.src_num = 2;
        let user = user_data(&[(0, 0), (1, 1), (2, 32), (3, 0)]);
        let program = [shr, endpgm(4)];
        assert_eq!(
            ScalarValue64 {
                lo: resolve(&program, 8, &user),
                hi: resolve(&program, 9, &user),
            }
            .full(),
            Some(1)
        );
    }

    #[test]
    fn s_bfe_u64_extracts_a_64_bit_field() {
        // offset 4, width 8 out of 0x…ff0 -> 0xff.
        let user = user_data(&[(0, 0x0000_0ff0), (1, 0), (2, 0x0008_0004), (3, 0)]);
        let mut bfe = ShaderInstruction {
            pc: 0,
            type_: T::SBfeU64,
            format: F::Sdst2Ssrc02Ssrc1,
            ..Default::default()
        };
        bfe.dst = sgpr(8, 2);
        bfe.src[0] = sgpr(0, 2);
        bfe.src[1] = sgpr(2, 1);
        bfe.src_num = 2;
        let program = [bfe, endpgm(4)];
        assert_eq!(
            ScalarValue64 {
                lo: resolve(&program, 8, &user),
                hi: resolve(&program, 9, &user),
            }
            .full(),
            Some(0xff)
        );
    }

    #[test]
    fn s_cselect_b64_follows_a_proven_scc() {
        let user = user_data(&[(0, 0xaa), (1, 0), (2, 0xbb), (3, 0)]);
        let mut cmp = ShaderInstruction {
            pc: 0,
            type_: T::SCmpLgU32,
            format: F::Ssrc0Ssrc1,
            ..Default::default()
        };
        cmp.src[0] = imm(1);
        cmp.src[1] = imm(1);
        cmp.src_num = 2;
        // 1 != 1 is false -> SCC false -> select src1 (0xbb).
        let program = [cmp, alu2_64(4, T::SCselectB64, 8, 0, 2), endpgm(8)];
        assert_eq!(resolve(&program, 8, &user), ScalarValue::Known(0xbb));
    }

    #[test]
    fn a_negative_inline_constant_sign_extends_into_a_64_bit_slot() {
        // s_and_b64 s[8:9], -1, s[0:1]  must leave s[0:1] unchanged, which only
        // works if the inline -1 fills both dwords.
        let user = user_data(&[(0, 0xdead_beef), (1, 0x1234_5678)]);
        let mut and64 = ShaderInstruction {
            pc: 0,
            type_: T::SAndB64,
            format: F::Sdst2Ssrc02Ssrc12,
            ..Default::default()
        };
        and64.dst = sgpr(8, 2);
        and64.src[0] = imm((-1i32) as u32);
        and64.src[1] = sgpr(0, 2);
        and64.src_num = 2;
        let program = [and64, endpgm(4)];
        assert_eq!(resolve(&program, 8, &user), ScalarValue::Known(0xdead_beef));
        assert_eq!(resolve(&program, 9, &user), ScalarValue::Known(0x1234_5678));
    }

    #[test]
    fn a_32_bit_literal_zero_extends_into_a_64_bit_slot() {
        // s_or_b64 s[8:9], 0xffffffff, 0  -> hi must be 0, not 0xffffffff.
        let mut or64 = ShaderInstruction {
            pc: 0,
            type_: T::SOrB64,
            format: F::Sdst2Ssrc02Ssrc12,
            ..Default::default()
        };
        or64.dst = sgpr(8, 2);
        or64.src[0] = lit(0xffff_ffff);
        or64.src[1] = imm(0);
        or64.src_num = 2;
        let program = [or64, endpgm(4)];
        let user = user_data(&[(0, 0)]);
        assert_eq!(resolve(&program, 8, &user), ScalarValue::Known(0xffff_ffff));
        assert_eq!(resolve(&program, 9, &user), ScalarValue::Known(0));
    }

    #[test]
    fn s_getpc_b64_folds_to_the_following_instruction_address() {
        let mut getpc = ShaderInstruction {
            pc: 0,
            type_: T::SGetpcB64,
            format: F::Sdst2,
            ..Default::default()
        };
        getpc.dst = sgpr(8, 2);
        getpc.src[0] = lit(0x0040_0004);
        getpc.src[1] = lit(0x0000_0002);
        getpc.src_num = 2;
        let program = [getpc, endpgm(4)];
        let user = user_data(&[(0, 0)]);
        assert_eq!(resolve(&program, 8, &user), ScalarValue::Known(0x0040_0004));
        assert_eq!(resolve(&program, 9, &user), ScalarValue::Known(2));
    }

    // ---- compares ---------------------------------------------------------

    #[test]
    fn every_scalar_compare_sets_scc_from_proven_operands() {
        let cases: &[(ShaderInstructionType, u32, u32, bool)] = &[
            (T::SCmpEqI32, 5, 5, true),
            (T::SCmpLgI32, 5, 5, false),
            (T::SCmpGtI32, (-1i32) as u32, 0, false),
            (T::SCmpGeI32, 0, 0, true),
            (T::SCmpLtI32, (-1i32) as u32, 0, true),
            (T::SCmpLeI32, 1, 0, false),
            (T::SCmpEqU32, 7, 7, true),
            (T::SCmpLgU32, 7, 8, true),
            (T::SCmpGtU32, 0xffff_ffff, 0, true),
            (T::SCmpGeU32, 3, 4, false),
            (T::SCmpLtU32, 3, 4, true),
            (T::SCmpLeU32, 4, 4, true),
        ];
        for &(type_, a, b, expect) in cases {
            let mut cmp = ShaderInstruction {
                pc: 0,
                type_,
                format: F::Ssrc0Ssrc1,
                ..Default::default()
            };
            cmp.src[0] = lit(a);
            cmp.src[1] = lit(b);
            cmp.src_num = 2;
            let program = [cmp, endpgm(4)];
            let state = evaluate_before(&program, 1, None, 0).expect("straight line");
            assert_eq!(state.scc(), Some(expect), "{type_:?} {a:#x} vs {b:#x}");
        }
    }

    #[test]
    fn a_compare_with_an_unproven_operand_leaves_scc_unproven() {
        let user = user_data(&[(0, 1)]);
        let mut cmp = ShaderInstruction {
            pc: 0,
            type_: T::SCmpEqU32,
            format: F::Ssrc0Ssrc1,
            ..Default::default()
        };
        // s9 is beyond `count`.
        cmp.src[0] = sgpr(9, 1);
        cmp.src[1] = imm(0);
        cmp.src_num = 2;
        let program = [cmp, endpgm(4)];
        let state = evaluate_before(&program, 1, Some(&user), 0).expect("straight line");
        assert_eq!(state.scc(), None);
    }

    // ---- unknown propagation through real definitions ---------------------

    #[test]
    fn a_value_loaded_from_guest_memory_stays_unknown() {
        // s_load_dword s4, s[0:1], 0 ; s_add_u32 s5, s4, 16
        // s4 is whatever guest memory held at translate time — NOT a proven
        // dispatch constant — so s5 must not fold even though 16 is a literal.
        let user = user_data(&[(0, 0x0010_0000), (1, 0)]);
        let program = [
            sload(0, 4, 0),
            alu2(8, T::SAddU32, 5, sgpr(4, 1), imm(16)),
            endpgm(12),
        ];
        assert_eq!(
            resolve(&program, 5, &user),
            ScalarValue::Unknown,
            "a scalar-memory result must not be folded into an address"
        );
    }

    #[test]
    fn a_lane_dependent_definition_stays_unknown() {
        // `v_cmp_eq_u32 s[4:5], v0, v1` — a VOPC lane mask landing in an SGPR
        // pair. s4 is also a captured live-in slot, so the only thing that can
        // keep it from being folded is the kill.
        let user = user_data(&[(4, 0x1234), (5, 0x5678)]);
        let mut vcmp = ShaderInstruction {
            pc: 0,
            type_: T::VCmpEqU32,
            ..Default::default()
        };
        vcmp.dst = sgpr(4, 2);
        vcmp.src[0] = ShaderOperand {
            type_: O::Vgpr,
            register_id: 0,
            size: 1,
            ..Default::default()
        };
        vcmp.src[1] = ShaderOperand {
            type_: O::Vgpr,
            register_id: 1,
            size: 1,
            ..Default::default()
        };
        vcmp.src_num = 2;
        let program = [vcmp, endpgm(4)];
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Unknown);
        assert_eq!(resolve(&program, 5, &user), ScalarValue::Unknown);
    }

    #[test]
    fn an_unmodelled_definition_kills_its_whole_destination_span() {
        // A wide unmodelled write (an x4 scalar load into s[4:7]) must kill all
        // four registers, not only the first.
        let user = user_data(&[(0, 0x20_0000), (1, 0), (4, 1), (5, 2), (6, 3), (7, 4)]);
        let mut load = ShaderInstruction {
            pc: 0,
            type_: T::SLoadDwordx4,
            format: F::Sdst4SbaseSoffsetOffset,
            ..Default::default()
        };
        load.dst = sgpr(4, 4);
        load.src[0] = sgpr(0, 2);
        load.src[1] = imm(0);
        load.src[2] = imm(0);
        load.src_num = 3;
        let program = [load, endpgm(4)];
        for reg in 4..8 {
            assert_eq!(
                resolve(&program, reg, &user),
                ScalarValue::Unknown,
                "s{reg} was defined by an unmodelled load"
            );
        }
    }

    #[test]
    fn an_unmodelled_definition_also_invalidates_scc() {
        // s_cmp_eq_u32 proves SCC, then an unmodelled SGPR write may clobber
        // it; the evaluator must not keep the stale condition.
        let mut cmp = ShaderInstruction {
            pc: 0,
            type_: T::SCmpEqU32,
            format: F::Ssrc0Ssrc1,
            ..Default::default()
        };
        cmp.src[0] = imm(0);
        cmp.src[1] = imm(0);
        cmp.src_num = 2;
        let program = [cmp, sload(4, 4, 0), endpgm(8)];
        let state = evaluate_before(&program, 2, None, 0).expect("straight line");
        assert_eq!(state.scc(), None);
    }

    // ---- control flow -----------------------------------------------------

    fn branch(pc: u32, type_: ShaderInstructionType, target: u32) -> ShaderInstruction {
        let mut inst = ShaderInstruction {
            pc,
            type_,
            format: F::Label,
            ..Default::default()
        };
        inst.src[0] = ShaderOperand {
            type_: O::LiteralConstant,
            constant: ShaderConstant::from_i(target as i32 - pc as i32 - 4),
            size: 0,
            ..Default::default()
        };
        inst.src_num = 1;
        inst
    }

    #[test]
    fn a_decidable_conditional_branch_is_followed_exactly() {
        // s_cmp_eq_u32 0,0 (SCC=1); s_cbranch_scc1 -> pc 12 (skips the s_mov);
        // the skipped move must NOT be applied.
        let mut cmp = ShaderInstruction {
            pc: 0,
            type_: T::SCmpEqU32,
            format: F::Ssrc0Ssrc1,
            ..Default::default()
        };
        cmp.src[0] = imm(0);
        cmp.src[1] = imm(0);
        cmp.src_num = 2;
        let mut skipped = ShaderInstruction {
            pc: 8,
            type_: T::SMovB32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        skipped.dst = sgpr(4, 1);
        skipped.src[0] = lit(0xbad);
        skipped.src_num = 1;

        let program = [
            cmp,
            branch(4, T::SCbranchScc1, 12),
            skipped,
            alu2(12, T::SAddU32, 5, sgpr(0, 1), imm(1)),
            endpgm(16),
        ];
        let user = user_data(&[(0, 0x10), (4, 0x77)]);
        // s4 keeps its live-in: the branch was taken, so the move never ran.
        assert_eq!(resolve(&program, 4, &user), ScalarValue::Known(0x77));
        assert_eq!(resolve(&program, 5, &user), ScalarValue::Known(0x11));
    }

    #[test]
    fn an_undecidable_conditional_branch_is_refused_by_name() {
        // s_cbranch_execz: exec is never proven, so both successors are live.
        let program = [
            branch(0, T::SCbranchExecz, 8),
            alu2(4, T::SAddU32, 5, sgpr(0, 1), imm(1)),
            endpgm(8),
        ];
        let user = user_data(&[(0, 1)]);
        assert_eq!(
            evaluate_before(&program, 1, Some(&user), 0),
            Err(ScalarEvalRefusal::UndecidableBranch { pc: 0 })
        );
    }

    #[test]
    fn a_backward_branch_that_can_re_run_the_target_is_refused_as_a_loop() {
        // s_mov s4,0 ; TARGET s_add_u32 s5,s4,0 ; s_add_u32 s4,s4,16 ;
        // s_cbranch_scc1 back to TARGET. The target runs many times with
        // different s4, so no single snapshot describes it.
        let mut mov = ShaderInstruction {
            pc: 0,
            type_: T::SMovB32,
            format: F::SVdstSVsrc0,
            ..Default::default()
        };
        mov.dst = sgpr(4, 1);
        mov.src[0] = imm(0);
        mov.src_num = 1;
        let program = [
            mov,
            alu2(4, T::SAddU32, 5, sgpr(4, 1), imm(0)),
            alu2(8, T::SAddU32, 4, sgpr(4, 1), imm(16)),
            branch(12, T::SCbranchScc1, 4),
            endpgm(16),
        ];
        assert_eq!(
            evaluate_before(&program, 1, None, 0),
            Err(ScalarEvalRefusal::Loop {
                pc: 12,
                target_pc: 4
            })
        );
    }

    #[test]
    fn an_indirect_pc_write_before_the_target_is_refused() {
        let mut setpc = ShaderInstruction {
            pc: 0,
            type_: T::SSetpcB64,
            format: F::Saddr,
            ..Default::default()
        };
        setpc.src[0] = sgpr(0, 2);
        setpc.src_num = 1;
        let program = [setpc, alu2(4, T::SAddU32, 5, imm(1), imm(1)), endpgm(8)];
        assert_eq!(
            evaluate_before(&program, 1, None, 0),
            Err(ScalarEvalRefusal::IndirectBranch { pc: 0 })
        );
    }

    #[test]
    fn s_endpgm_before_the_target_makes_it_unreachable() {
        let program = [endpgm(0), alu2(4, T::SAddU32, 5, imm(1), imm(1))];
        assert_eq!(
            evaluate_before(&program, 1, None, 0),
            Err(ScalarEvalRefusal::Unreachable)
        );
    }

    #[test]
    fn an_out_of_range_target_index_is_refused_rather_than_panicking() {
        let program = [endpgm(0)];
        assert_eq!(
            evaluate_before(&program, 9, None, 0),
            Err(ScalarEvalRefusal::BadIndex)
        );
    }

    // ---- the never-written route -----------------------------------------

    #[test]
    fn a_never_written_register_resolves_even_through_undecidable_control_flow() {
        // s4 is a live-in nothing writes. The shader has an undecidable
        // s_cbranch_execz, which the walk would refuse — but the value of a
        // register with no definition anywhere does not depend on the path.
        let program = [
            branch(0, T::SCbranchExecz, 8),
            alu2(4, T::SAddU32, 9, imm(1), imm(1)),
            sload(8, 20, 0),
        ];
        let user = user_data(&[(4, 0x180)]);
        assert_eq!(
            resolve_sgpr_before(&program, 2, 4, Some(&user), 0),
            Ok(ScalarValue::Known(0x180))
        );
        // Same shader, a register that IS written: the walk runs and refuses.
        assert_eq!(
            resolve_sgpr_before(&program, 2, 9, Some(&user), 0),
            Err(ScalarEvalRefusal::UndecidableBranch { pc: 0 })
        );
    }

    #[test]
    fn a_never_written_register_outside_the_captured_user_data_is_unknown() {
        let program = [endpgm(0)];
        let user = user_data(&[(0, 1)]);
        // s40 > SGPRS_MAX and far beyond count.
        assert_eq!(
            resolve_sgpr_before(&program, 0, 40, Some(&user), 0),
            Ok(ScalarValue::Unknown)
        );
        // No user data at all (the PC-relative caller).
        assert_eq!(
            resolve_sgpr_before(&program, 0, 0, None, 0),
            Ok(ScalarValue::Unknown)
        );
    }

    #[test]
    fn the_multi_step_chain_that_motivated_this_module_folds() {
        // The measured shape: a live-in index scaled and biased into a byte
        // offset. s0 = 3 -> s4 = 3 << 5 = 0x60 -> s5 = 0x60 | 0 -> s6 = 0x60 + 0x40.
        let user = user_data(&[(0, 3), (1, 0x40)]);
        let program = [
            alu2(0, T::SLshlB32, 4, sgpr(0, 1), imm(5)),
            alu2(4, T::SAndB32, 5, sgpr(4, 1), lit(0x0000_ffff)),
            alu2(12, T::SAddU32, 6, sgpr(5, 1), sgpr(1, 1)),
            endpgm(16),
        ];
        assert_eq!(resolve(&program, 6, &user), ScalarValue::Known(0xa0));
    }

    #[test]
    fn a_named_scalar_soffset_is_proven_only_after_a_modelled_definition() {
        // GTA's measured pixel shader computes an S_BUFFER_LOAD offset as
        // `vcc_lo = s6 << 4`. Named scalar registers are not user-data live-ins,
        // but a deterministic definition before the load is safe to fold.
        let user = user_data(&[(6, 3)]);
        let mut scale = alu2(0, T::SLshlB32, 0, sgpr(6, 1), imm(4));
        scale.dst = named(O::VccLo);
        let program = [scale, endpgm(4)];
        let vcc_lo = named(O::VccLo);

        assert_eq!(
            resolve_scalar_operand_before(&program, 1, &vcc_lo, Some(&user), 0),
            Ok(ScalarValue::Known(48))
        );

        let no_definition = [endpgm(0)];
        assert_eq!(
            resolve_scalar_operand_before(&no_definition, 0, &vcc_lo, Some(&user), 0),
            Ok(ScalarValue::Unknown)
        );
    }
}
