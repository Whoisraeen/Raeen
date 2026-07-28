//! GCN shader data model, ported from Kyty (MIT (c) InoriRus).
//!
//! Kyty sources:
//! - `emulator/include/Emulator/Graphics/Shader.h` (data model)
//! - `emulator/src/Graphics/Shader.cpp` (`operand_to_str` L117,
//!   `operand_array_to_str` L170, `dbg_fmt_to_str` L222, `dbg_fmt_print` L282,
//!   `DbgInstructionToStr` L397, `DbgDump` L410, `ReadBlock` L474,
//!   `ReadIntructions` L509)
//!
//! C++ `type` fields are `type_` in Rust (`type` is a keyword). Kyty hard-EXIT
//! assertions in the debug printers are replaced by graceful `"???"` output —
//! library code must never panic on arbitrary decoded data.

use std::fmt::Write as _;

/// Kyty: Shader.h `ShaderType` (L24).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ShaderType {
    #[default]
    Unknown,
    Vertex,
    Pixel,
    Fetch,
    Compute,
}

/// Kyty: Shader.h `ShaderInstructionType` (L33-233). Complete list — the
/// SPIR-V recompiler batch dispatches on these even where the parser cannot
/// reach them yet.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShaderInstructionType {
    #[default]
    Unknown,

    BufferLoadDword,
    /// MUBUF 0x0d: two-dword raw load. Kyty leaves it `KYTY_NI`; measured in
    /// ASTRO.BOT scene compute (raw 0xe0342000, idxen).
    BufferLoadDwordX2,
    /// MUBUF 0x0f: three-dword raw load. Kyty leaves it `KYTY_NI`; measured
    /// in ASTRO.BOT scene compute (raws 0xe03c2074/0xe03c2034, idxen with a
    /// nonzero immediate offset). 116 dispatches in the measured window.
    BufferLoadDwordX3,
    BufferLoadDwordX4,
    /// MUBUF 0x08: single byte load, zero-extended into the VGPR. Kyty
    /// leaves it `KYTY_NI`; measured on ASTRO.BOT scene compute (raw
    /// 0xe02020c0, idxen with immediate offset 0xc0; 58 dispatches/run).
    /// The recompiler loads the containing dword and extracts the byte at
    /// `(byte_addr & 3) * 8` — the byte address is NOT pre-divided by 4.
    BufferLoadUbyte,
    BufferLoadFormatX,
    BufferLoadFormatXy,
    BufferLoadFormatXyz,
    BufferLoadFormatXyzw,
    BufferStoreDword,
    /// MUBUF 0x1d: two-dword raw store. Kyty leaves it `KYTY_NI`; measured in
    /// ASTRO.BOT scene compute (0x500757800). Same flexible addressing quartet
    /// as [`BufferStoreDwordX4`]; the shared `buffer_store_dwordxn` helper
    /// writes `n` consecutive dwords.
    BufferStoreDwordX2,
    /// MUBUF 0x1e: four-dword raw store. Kyty leaves it `KYTY_NI`; measured
    /// in ASTRO.BOT scene compute (raw 0xe0780000).
    BufferStoreDwordX4,
    BufferStoreFormatX,
    BufferStoreFormatXy,
    /// MUBUF 0x06: formatted 3-channel store. Kyty leaves it `KYTY_NI`
    /// (ShaderParse.cpp L2629); measured in ASTRO.BOT scene compute.
    BufferStoreFormatXyz,
    /// MUBUF 0x07: formatted 4-channel store. Kyty leaves it `KYTY_NI`
    /// (ShaderParse.cpp L2630); the single most frequent ASTRO.BOT shader
    /// failure (925 dispatches in the measured 30s window).
    BufferStoreFormatXyzw,

    // --- FLAT-class memory (FLAT / GLOBAL segments, encoding 0x37) ---
    //
    // Beyond Kyty (its SI/GNM parser has no FLAT class): a GFX10/RDNA2
    // FLAT-class load/store that addresses guest memory *directly* by a
    // complete 64-bit pointer rather than through a bound V# descriptor.
    // Ported from SharpEmu PR #587 (`Gen5ShaderTranslator.DecodeFlat`,
    // GPL-2.0). The FLAT segment carries the whole 64-bit address in the VGPR
    // pair `(addr, addr+1)`; the GLOBAL segment adds a 32-bit VGPR offset to an
    // SGPR base pair. `ShaderInstruction::uses_flat_address` records which form
    // this instruction decoded to (true = address is a VGPR pair). Unblocks
    // GTA V Enhanced's frontend / legal-text shaders and any title whose
    // compiler emits flat/global addressing.
    /// FLAT 0x08 `flat_load_ubyte` — single zero-extended byte load.
    FlatLoadUbyte,
    /// FLAT 0x0c `flat_load_dword` — one 32-bit dword load.
    FlatLoadDword,
    /// FLAT 0x0d `flat_load_dwordx2` — two consecutive dwords.
    FlatLoadDwordX2,
    /// FLAT 0x0f `flat_load_dwordx3` — three consecutive dwords.
    FlatLoadDwordX3,
    /// FLAT 0x0e `flat_load_dwordx4` — four consecutive dwords.
    FlatLoadDwordX4,
    /// FLAT 0x1c `flat_store_dword` — one dword store (data in `dst`).
    FlatStoreDword,
    /// FLAT 0x1d `flat_store_dwordx2` — two consecutive dword store.
    FlatStoreDwordX2,
    /// FLAT 0x1e `flat_store_dwordx4` — four consecutive dword store.
    FlatStoreDwordX4,

    /// DS 0x00: LDS atomic dword add without return (RDNA2 ISA `DS_ADD_U32`).
    /// Kyty leaves the whole DS family except append/consume `KYTY_NI`;
    /// measured on ASTRO.BOT scene compute (raw 0xd8000514, 58 skips/run).
    /// Lowered to an exec-guarded `OpAtomicIAdd` on the `%lds` Workgroup
    /// array at `(addr + offset) >> 2`.
    DsAddU32,
    /// DS 0x20: LDS atomic dword add returning the old value
    /// (`vdst = lds[a]; lds[a] += data`). Measured on ASTRO.BOT after mixed
    /// storage-image routing advanced tiled-lighting compute.
    DsAddRtnU32,
    DsAppend,
    DsConsume,
    /// DS 0x37: two independent LDS dword reads at `addr + offset0*4` and
    /// `addr + offset1*4` into `vdst`/`vdst+1` (offsets are in DWORD units
    /// for the read2/write2 forms, unlike the byte offset of the single
    /// forms — RDNA2 ISA `DS_READ2_B32`). Kyty leaves the whole DS family
    /// except append/consume `KYTY_NI`; measured on ASTRO.BOT scene compute
    /// (raw 0xd8dc0100).
    DsRead2B32,
    /// DS 0x76: two CONSECUTIVE LDS dwords read at the single 16-bit byte
    /// offset into `vdst`/`vdst+1` (RDNA2 ISA `DS_READ_B64`). Parsed into the
    /// same `Vdst2Vsrc0Vsrc1Vsrc2` shape as `DsRead2B32` with the second
    /// offset literal set to `offset + 4`, so one recompile body serves both.
    /// Kyty `KYTY_NI`; measured on ASTRO.BOT scene compute (raw 0xd9d80000,
    /// 58 dispatches/run).
    DsReadB64,
    /// DS 0xff: four CONSECUTIVE LDS dwords read at the single 16-bit byte
    /// offset into `vdst..vdst+3` (RDNA2 ISA `DS_READ_B128`). Kyty
    /// `KYTY_NI`; measured on ASTRO.BOT scene compute (58 dispatches/run).
    /// Extends the b64 model: dst = 4 consecutive VGPRs, src0 = address,
    /// src1 = the byte offset literal (dword k reads at `offset + 4k`).
    DsReadB128,
    /// DS 0xfe: three CONSECUTIVE LDS dwords read at the single 16-bit byte
    /// offset into `vdst..vdst+2` (RDNA2 ISA `DS_READ_B96`). Kyty
    /// `KYTY_NI`; measured on ASTRO.BOT scene compute (raw 0xdbf80550,
    /// 58 dispatches/run). Same model as `DsReadB128` with three dwords.
    DsReadB96,
    /// DS 0x36: LDS (workgroup-shared) dword read. Kyty leaves it `KYTY_NI`
    /// (only the GDS append/consume pair is implemented upstream); lowered to
    /// an `OpLoad` from the `%lds` Workgroup array. Read twin of `DsWriteB32`.
    DsReadB32,
    /// DS 0x0d: LDS (workgroup-shared) dword write — measured in ASTRO.BOT
    /// scene compute (raw 0xd8340000). Lowered to an `OpStore` into the
    /// `%lds` Workgroup array at `(addr + offset) >> 2`.
    DsWriteB32,
    /// DS 0x2d: LDS atomic write-exchange, returning the OLD value — measured
    /// in ASTRO.BOT tiled-lighting compute (raw 0xd8b40510). `vdst = lds[a];
    /// lds[a] = data`. Lowered to `OpAtomicExchange` on the `%lds` Workgroup
    /// array (exec-guarded like `DsAddU32`, old value written to `vdst`).
    DsWrxchgRtnB32,
    /// DS 0xde: three consecutive LDS dwords stored from `data0..data0+2` at
    /// the 16-bit byte offset (RDNA2 ISA `DS_WRITE_B96`). Kyty `KYTY_NI`;
    /// measured on ASTRO.BOT scene compute.
    DsWriteB96,
    /// DS 0xdf: four consecutive LDS dwords stored from `data0..data0+3` at
    /// the 16-bit byte offset (RDNA2 ISA `DS_WRITE_B128`). Kyty `KYTY_NI`;
    /// measured on ASTRO.BOT scene compute (raw 0xdb7c0000).
    DsWriteB128,
    Exp,
    /// MIMG 0x47: four-texel gather of a single channel at an implicit zero
    /// LOD (RDNA2 ISA `IMAGE_GATHER4_LZ`). Kyty `KYTY_NI`; measured on
    /// ASTRO.BOT scene compute (raw 0xf11c0108, dmask 1). The uploaded
    /// images carry exactly one mip, so a plain `OpImageGather` (which
    /// samples the base level) is the LZ semantic.
    ImageGather4Lz,
    /// MIMG 0x0e: query texture dimensions/mip information.
    ImageGetResinfo,
    ImageLoad,
    ImageSample,
    /// MIMG 0x24: sampled image read with an explicit LOD supplied after the
    /// dimensional coordinate tuple in VADDR. Measured on ASTRO.BOT scene
    /// compute (raw 0xf0900718).
    ImageSampleL,
    /// MIMG 0x2f: comparison sample with an explicit zero LOD.
    ImageSampleCLz,
    ImageSampleLz,
    ImageSampleLzO,
    ImageStore,
    ImageStoreMip,
    SAddcU32,
    SAddI32,
    SAddU32,
    SAndB32,
    SAndB64,
    /// RDNA2 (`next_gen`) SOP1 0x37: `sdst = exec; exec = ~ssrc0 & exec`
    /// (save-exec, negating the first operand). SharpEmu Gen5 decodes this as
    /// `SAndn1SaveexecB64` (Gen5ShaderTranslator.cs L710); the `andn2` sibling
    /// (0x27) negates the second operand instead. Measured in ASTRO.BOT's
    /// scene-composite compute shader (0x555f4f500, divergent-flow prologue).
    SAndn1SaveexecB64,
    SAndn2B64,
    SAndSaveexecB64,
    /// SOPP 0x0a: workgroup execution + LDS memory barrier. Kyty leaves it
    /// `KYTY_NI`; required by the `ds_write_b32`/`ds_read_b32` LDS pairs.
    SBarrier,
    SBfeU32,
    SBfeU64,
    SBfmB32,
    SBranch,
    SBrevB32,
    SBufferLoadDword,
    SBufferLoadDwordx16,
    SBufferLoadDwordx2,
    SBufferLoadDwordx4,
    SBufferLoadDwordx8,
    SCbranchExecz,
    SCbranchScc0,
    SCbranchScc1,
    SCbranchVccz,
    SCbranchVccnz,
    SCmpEqI32,
    SCmpEqU32,
    SCmpGeI32,
    SCmpGeU32,
    SCmpGtI32,
    SCmpGtU32,
    SCmpLeI32,
    SCmpLeU32,
    SCmpLgI32,
    SCmpLgU32,
    SCmpLtI32,
    SCmpLtU32,
    SCselectB32,
    SCselectB64,
    SEndpgm,
    /// SOP1 0x1f: write the absolute address of the following instruction.
    SGetpcB64,
    SInstPrefetch,
    SLoadDword,
    SLoadDwordx2,
    SLoadDwordx4,
    SLoadDwordx8,
    SLoadDwordx16,
    SLshl4AddU32,
    SLshlB32,
    SLshrB32,
    SLshlB64,
    SLshrB64,
    SMovB32,
    SMovB64,
    SMovkI32,
    SMulHiU32,
    SMulI32,
    SMulkI32,
    SNandB64,
    SNop,
    SNorB64,
    SNotB64,
    SOrB32,
    SOrB64,
    SOrn2B64,
    /// SOP1 0x28: `sdst = exec; exec = ssrc0 | ~exec; scc = (exec != 0)`. The
    /// ORN2 sibling of `SAndSaveexecB64`; measured in ASTRO.BOT scene compute.
    SOrn2SaveexecB64,
    /// SOP2 0x32: pack the low 16 bits of each source into one dword.
    SPackLlB32B16,
    SSendmsg,
    /// RDNA2 SOPK opcode 1: code-object version marker; no execution effect.
    SVersion,
    SSetpcB64,
    SSwappcB64,
    SSubI32,
    SSubU32,
    SWaitcnt,
    SWqmB64,
    SXnorB64,
    SXorB64,
    TBufferLoadFormatX,
    TBufferLoadFormatXyzw,
    VAddF32,
    /// RDNA2 (`next_gen`) VOP3 0x36d: `vdst = vsrc0 + vsrc1 + vsrc2` (32-bit,
    /// carry-less). Kyty names it `KYTY_NI` (ShaderParse.cpp L2112); shadPS4:
    /// `V_ADD3_U32 = 877` (== 0x36d). Measured in ASTRO.BOT scene compute.
    VAdd3U32,
    VAddI32,
    /// RDNA2 (`next_gen`) v_add_co_ci_u32 — add with carry-in and carry-out:
    /// `vdst = src0 + src1 + carry_in; carry_out -> sdst`. Two encodings feed
    /// this one type: the plain VOP2 form (opcode 0x28, carry in/out both via
    /// VCC) and the VOP3B form (opcode 0x128, carry-in = src2, carry-out =
    /// sdst). SharpEmu Gen5 names the VOP3B form `VAddCoCiU32`
    /// (Gen5ShaderTranslator.cs L1094, IsVop3BOpcode L1163) and lowers it in
    /// `EmitAddWithCarry` (Gen5SpirvTranslator.Alu.cs L3396). The VOP3B sdst
    /// field (bits [14:8]) overlaps the VOP3A op_sel bits [14:11], so before
    /// this type existed the decoder misread the carry-out SGPR as `op_sel !=
    /// 0` and refused the whole shader. Measured in ASTRO.BOT's scene-composite
    /// compute shader (0x555f4f500).
    VAddCoCiU32,
    /// RDNA2 (`next_gen`) VOP2 0x25: carry-less `vdst = vsrc0 + vsrc1`
    /// (replaces GCN's carry-writing v_add_i32 in the same encoding slot).
    VAddNcU32,
    VAndB32,
    VAshrI32,
    VAshrrevI32,
    VBcntU32B32,
    /// VOP3 0x149: signed bitfield extract — `vdst = SignExtend((vsrc0 >>
    /// vsrc1[4:0])[vsrc2[4:0]-1 : 0])`. Signed twin of `VBfeU32`; measured
    /// on ASTRO.BOT scene compute (58 dispatches/run).
    VBfeI32,
    VBfeU32,
    /// VOP3 0x14a: bitfield insert — `vdst = (vsrc0 & vsrc1) | (~vsrc0 &
    /// vsrc2)`. Kyty `KYTY_NI`; measured on ASTRO.BOT scene compute
    /// (58 dispatches/run).
    VBfiB32,
    VBfmB32,
    VBfrevB32,
    VCeilF32,
    VCmpEqF32,
    VCmpEqI32,
    VCmpEqU32,
    VCmpFF32,
    VCmpFI32,
    VCmpFU32,
    VCmpGeF32,
    VCmpGeI32,
    VCmpGeU32,
    VCmpGtF32,
    VCmpGtI32,
    VCmpGtU32,
    /// GFX10 VOPC/VOP3 0xe4: unsigned 64-bit greater-than. Sources are
    /// adjacent low/high dwords and the scalar-mask destination is a pair.
    /// Measured in ASTRO.BOT scene compute.
    VCmpGtU64,
    VCmpLeF32,
    VCmpLeI32,
    VCmpLeU32,
    VCmpLgF32,
    VCmpLtF32,
    VCmpLtI32,
    VCmpLtU32,
    VCmpNeI32,
    VCmpNeqF32,
    VCmpNeU32,
    VCmpNgeF32,
    VCmpNgtF32,
    VCmpNleF32,
    VCmpNlgF32,
    VCmpNltF32,
    VCmpOF32,
    VCmpTI32,
    VCmpTruF32,
    VCmpTU32,
    VCmpUF32,
    VCmpxEqF32,
    VCmpxEqU32,
    /// VOPC 0x16: `exec/smask = vsrc0 >= vsrc1` (ordered). Exec-writing
    /// sibling of `VCmpGeF32`; measured in ASTRO.BOT scene CS.
    VCmpxGeF32,
    /// VOPC 0x19: `exec/smask = !(vsrc0 >= vsrc1)` — the UNORDERED `<`
    /// (NaN → true). Exec-writing sibling of `VCmpNgeF32`; measured in
    /// ASTRO.BOT tiled-lighting compute (raw 0x7c32d4f9).
    VCmpxNgeF32,
    VCmpxGeU32,
    VCmpxGtF32,
    VCmpxGtU32,
    VCmpxEqI32,
    VCmpxGeI32,
    VCmpxGtI32,
    /// VOPC 0x13: `exec/smask = vsrc0 <= vsrc1` (ordered). Exec-writing
    /// sibling of `VCmpLeF32`; measured in ASTRO.BOT scene CS
    /// (58 dispatches/run).
    VCmpxLeF32,
    VCmpxLeI32,
    /// VOPC 0xd3: `exec/smask = vsrc0 <= vsrc1` (unsigned). Measured after
    /// ASTRO.BOT's mixed-storage-image shader advanced to tiled lighting.
    VCmpxLeU32,
    VCmpxLtF32,
    VCmpxLtI32,
    VCmpxLtU32,
    VCmpxNeI32,
    VCmpxNeqF32,
    /// VOPC 0x1c: `exec/smask = !(vsrc0 <= vsrc1)` (unordered >, NaN→true).
    /// Exec-writing sibling of `VCmpNleF32`; measured in ASTRO.BOT scene CS.
    VCmpxNleF32,
    /// VOPC 0x1e: `exec/smask = !(vsrc0 < vsrc1)` (unordered ≥, NaN→true). The
    /// exec-writing sibling of `VCmpNltF32`; measured in ASTRO.BOT scene CS.
    VCmpxNltF32,
    VCmpxNeU32,
    VCndmaskB32,
    VCosF32,
    VCvtF32F16,
    VCvtF32I32,
    VCvtF32U32,
    VCvtF32Ubyte0,
    VCvtF32Ubyte1,
    VCvtF32Ubyte2,
    VCvtF32Ubyte3,
    /// VOP1 0x8: `vdst = (int)vsrc0` (float→signed int). The signed sibling
    /// of `VCvtU32F32`, measured in Minecraft's menu CS.
    /// RDNA/GCN cubemap-coordinate helpers (VOP3 0x144-0x147). Together they
    /// turn a 3D direction (x=src0, y=src1, z=src2) into a cube face id, the
    /// S/T face coordinates, and the major-axis divisor. Formulas ported from
    /// shadPS4 (`vector_alu.cpp` V_CUBE*_F32 + `SelectCubeResult`).
    VCubeIdF32,
    VCubeScF32,
    VCubeTcF32,
    VCubeMaF32,
    VCvtI32F32,
    /// VOP1 0xd: `vdst = (int)floor(vsrc0)` (float→signed int, rounding toward
    /// −∞). The floor sibling of `VCvtI32F32` (which truncates toward zero);
    /// measured in ASTRO.BOT's scene compute shaders.
    VCvtFlrI32F32,
    VCvtPkrtzF16F32,
    VCvtU32F32,
    VExpF32,
    VFloorF32,
    VFmaF32,
    /// VOP3P 0x20 (`v_fma_mix_f32`, `v_mad_mix_f32` on gfx9). A SINGLE f32
    /// `fma(a, b, c)` whose three sources are each read independently as
    /// either a full f32 register/constant or one f16 half widened to f32 —
    /// selected per operand by `op_sel_hi` (read as f16 when set) and `op_sel`
    /// (which half). Not a packed op despite the VOP3P encoding.
    ///
    /// Beyond Kyty (SharpEmu PR #466 `3574a3b`): the whole VOP3P encoding was
    /// undecoded, so every shader containing one was DROPPED at
    /// `UnknownEncoding` — the mix ops are what Unity HDR pixel shaders use to
    /// combine half-precision inputs.
    VFmaMixF32,
    /// VOP3P 0x21 (`v_fma_mixlo_f16`). Same f32 `fma` as [`Self::VFmaMixF32`],
    /// then narrowed to f16 and merged into the LOW 16 bits of `vdst`, leaving
    /// the high half intact.
    VFmaMixloF16,
    /// VOP3P 0x22 (`v_fma_mixhi_f16`). The [`Self::VFmaMixloF16`] sibling that
    /// writes the HIGH 16 bits of `vdst`.
    VFmaMixhiF16,
    VFractF32,
    VInterpMovF32,
    VInterpP1F32,
    VInterpP2F32,
    VLogF32,
    /// RDNA2 (`next_gen`) VOP3 0x346: `vdst = (vsrc0 << vsrc1[4:0]) + vsrc2`.
    /// Not in Kyty's GCN table — first RDNA2-only instruction, added for the
    /// Minecraft menu CS.
    VAndOrB32,
    VLshlAddU32,
    VLshlOrU32,
    VOr3U32,
    VLshlB32,
    VLshlrevB32,
    VLshrB32,
    VLshrrevB32,
    VMacF32,
    VMadakF32,
    VMadF32,
    VMadmkF32,
    VMadU32U24,
    /// RDNA2 (`next_gen`) VOP3B 0x176 `v_mad_u64_u32`: widening
    /// multiply-accumulate `vdst.u64 = src0.u32 * src1.u32 + src2.u64`, with the
    /// 64-bit add's carry-out written to the `sdst` mask (bits [14:8], the same
    /// VOP3B field that overlaps VOP3A op_sel — see [`VAddCoCiU32`]). SharpEmu
    /// Gen5 names it in `IsVop3BOpcode` (Gen5ShaderTranslator.cs L1163) and
    /// lowers it via the mul-hi/lo + add-with-carry idiom. Measured in
    /// ASTRO.BOT's scene-composite compute shader (0x555f4f500) — the sole
    /// remaining parse wall on that shader after the op_sel gate landed.
    VMadU64U32,
    VMax3F32,
    VMaxF32,
    /// RDNA2 VOP2 0x12 `v_max_i32` — signed integer max (`GLSL SMax`).
    VMaxI32,
    /// RDNA2 VOP2 0x14 `v_max_u32` — unsigned integer max (`GLSL UMax`).
    VMaxU32,
    VMbcntHiU32B32,
    VMbcntLoU32B32,
    VMed3F32,
    VMin3F32,
    VMinF32,
    /// RDNA2 VOP2 0x11 `v_min_i32` — signed integer min (`GLSL SMin`).
    VMinI32,
    /// RDNA2 VOP2 0x13 `v_min_u32` — unsigned integer min (`GLSL UMin`).
    /// Measured in ASTRO.BOT scene-composite compute shader 0x555f4f500.
    VMinU32,
    VMovB32,
    VMulF32,
    VMulHiU32,
    VMulLoI32,
    VMulLoU32,
    VMulU32U24,
    VNotB32,
    VOrB32,
    /// VOP3P 0x0e `v_pk_fma_f16`: two independent f16 `fma`s, one per packed
    /// 16-bit lane of the destination. Beyond Kyty (SharpEmu PR #420
    /// `3005bab`). See [`Vop3pControl`] for the per-lane half select/negate.
    VPkFmaF16,
    /// VOP3P 0x0f `v_pk_add_f16`.
    VPkAddF16,
    /// VOP3P 0x10 `v_pk_mul_f16`.
    VPkMulF16,
    /// VOP3P 0x11 `v_pk_min_f16` (`fminnum_like`: a NaN operand yields the
    /// other).
    VPkMinF16,
    /// VOP3P 0x12 `v_pk_max_f16` (`fmaxnum_like`).
    VPkMaxF16,
    VRcpF32,
    /// VOP1 0x2b / VOP3 0x1ab `v_rcp_iflag_f32`: reciprocal whose only
    /// difference from `v_rcp_f32` is raising the integer-division-by-zero
    /// TRAP flag on 0/denorm inputs. Exceptions are not modelled, so the
    /// arithmetic lowering is identical (1.0 / x).
    VRcpIflagF32,
    VRndneF32,
    VRsqF32,
    VSadU32,
    VSinF32,
    VSqrtF32,
    VSubF32,
    VSubI32,
    /// RDNA2 VOP2 0x29 `v_subb_u32`: `src0 - src1 - VCC`, with the unsigned
    /// borrow-out written back to VCC.
    VSubbU32,
    /// RDNA2 VOP2 0x2a `v_subbrev_u32`: reverse subtract with borrow,
    /// `src1 - src0 - VCC`, with the unsigned borrow-out written back to VCC.
    VSubbrevU32,
    /// RDNA2 (`next_gen`) VOP2 0x26: carry-less `vdst = vsrc0 - vsrc1`.
    VSubNcU32,
    VSubrevF32,
    VSubrevI32,
    /// RDNA2 (`next_gen`) VOP2 0x27: carry-less `vdst = vsrc1 - vsrc0`.
    VSubrevNcU32,
    VTruncF32,
    /// RDNA2 (`next_gen`) VOP2 0x1e: `vdst = ~(vsrc0 ^ vsrc1)` (bitwise XNOR).
    /// Replaces GCN's v_bfm_b32 in this VOP2 slot; SharpEmu Gen5 lowers it as
    /// NOT(XOR). Measured in ASTRO.BOT's scene-composite compute shader.
    VXnorB32,
    VXorB32,

    FetchX,
    FetchXy,
    FetchXyz,
    FetchXyzw,

    ZMax,
}

/// Kyty: Shader.h namespace `ShaderInstructionFormat` (L235-359).
///
/// A `Format` is a u64-packed string of `FormatByte` tokens (low byte =
/// last-printed operand). It is both the disassembly spec (see
/// [`ShaderCode::dbg_instruction_to_str`]) and the recompiler dispatch key —
/// keep the packed-u64 mechanism intact.
pub mod shader_instruction_format {
    // FormatByte tokens — Kyty: Shader.h `enum FormatByte` (L237-291).
    // Kyty spells these U/N/D/../DmaskF/Gds; upper-snake per Rust const style.
    pub const U: u64 = 0;
    pub const N: u64 = 1;
    /// operand_to_str(inst.dst)
    pub const D: u64 = 2;
    /// operand_to_str(inst.dst2)
    pub const D2: u64 = 3;
    /// operand_to_str(inst.src[0])
    pub const S0: u64 = 4;
    /// operand_to_str(inst.src[1])
    pub const S1: u64 = 5;
    /// operand_to_str(inst.src[2])
    pub const S2: u64 = 6;
    /// operand_to_str(inst.src[3])
    pub const S3: u64 = 7;
    pub const DA2: u64 = 8;
    pub const DA3: u64 = 9;
    pub const DA4: u64 = 10;
    pub const DA8: u64 = 11;
    pub const DA16: u64 = 12;
    pub const D2A2: u64 = 13;
    pub const D2A3: u64 = 14;
    pub const D2A4: u64 = 15;
    pub const S0A2: u64 = 16;
    pub const S0A3: u64 = 17;
    pub const S0A4: u64 = 18;
    pub const S1A2: u64 = 19;
    pub const S1A3: u64 = 20;
    pub const S1A4: u64 = 21;
    pub const S1A8: u64 = 22;
    pub const S2A2: u64 = 23;
    pub const S2A3: u64 = 24;
    pub const S2A4: u64 = 25;
    /// attr%u.%u <- inst.src[1].constant.u, inst.src[2].constant.u
    pub const ATTR: u64 = 26;
    pub const IDXEN: u64 = 27;
    pub const OFFEN: u64 = 28;
    pub const FLOAT1: u64 = 29;
    pub const FLOAT4: u64 = 30;
    pub const POS0: u64 = 31;
    pub const DONE: u64 = 32;
    pub const PARAM0: u64 = 33;
    pub const PARAM1: u64 = 34;
    pub const PARAM2: u64 = 35;
    pub const PARAM3: u64 = 36;
    pub const PARAM4: u64 = 37;
    pub const MRT0: u64 = 38;
    pub const PRIM: u64 = 39;
    pub const OFF: u64 = 40;
    /// EXP target 9 — the `null` export target. Beyond Kyty, which EXITs on it.
    /// A pixel shader that writes no colour (depth-only, or one whose only
    /// colour path is `discard`) still has to terminate its export sequence,
    /// and does so with `exp null off,off,off,off done vm`.
    pub const NULL_TGT: u64 = 56;
    pub const COMPR: u64 = 41;
    pub const VM: u64 = 42;
    /// label_%u
    pub const L: u64 = 43;
    pub const DMASK_F: u64 = 44;
    pub const DMASK_7: u64 = 45;
    pub const DMASK_1: u64 = 46;
    pub const DMASK_8: u64 = 47;
    pub const DMASK_3: u64 = 48;
    pub const DMASK_5: u64 = 49;
    pub const DMASK_9: u64 = 50;
    pub const GDS: u64 = 51;
    // Beyond Kyty (upstream has no exp targets 0x0d-0x0f and no dmask:0x2
    // MIMG form) — added for the ASTRO.BOT Gen5 shader batch.
    pub const POS1: u64 = 52;
    pub const POS2: u64 = 53;
    pub const POS3: u64 = 54;
    pub const DMASK_2: u64 = 55;
    /// Beyond Kyty: two-channel ZW image-load mask. Measured on four
    /// ASTRO.BOT scene compute shaders after the mixed-storage path advanced.
    pub const DMASK_C: u64 = 57;
    /// Beyond Kyty: single-channel Z mask. Only reachable through the gather
    /// family, where the dmask names the ONE channel gathered rather than a
    /// destination-component subset (see [`Format::Vdata4Vaddr3StSsDmask4`]).
    pub const DMASK_4: u64 = 58;

    /// Kyty: Shader.h `FormatDefine` (L293). Packs FormatByte tokens into a
    /// u64, first token in the highest-used byte.
    #[must_use]
    pub const fn format_define(f: &[u64]) -> u64 {
        let mut r: u64 = 0;
        let mut i = 0;
        while i < f.len() {
            r = (r << 8) | f[i];
            i += 1;
        }
        r
    }

    /// Kyty: Shader.h `enum Format` (L303-357).
    #[repr(u64)]
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Format {
        #[default]
        Unknown = format_define(&[U]),
        Empty = format_define(&[N]),
        Imm = format_define(&[S0]),
        Label = format_define(&[L]),
        Mrt0OffOffComprVmDone = format_define(&[MRT0, OFF, OFF, COMPR, VM, DONE]),
        Mrt0Vsrc0Vsrc1ComprVmDone = format_define(&[MRT0, S0, S1, COMPR, VM, DONE]),
        Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone = format_define(&[MRT0, S0, S1, S2, S3, VM, DONE]),
        Param0Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM0, S0, S1, S2, S3]),
        Param1Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM1, S0, S1, S2, S3]),
        Param2Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM2, S0, S1, S2, S3]),
        Param3Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM3, S0, S1, S2, S3]),
        Param4Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM4, S0, S1, S2, S3]),
        Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done = format_define(&[POS0, S0, S1, S2, S3, DONE]),
        // Beyond Kyty: auxiliary position exports (exp targets 0x0d-0x0f).
        // These carry clip/cull distances or point size as configured by
        // PA_CL_VS_OUT_CNTL (shadPS4 `ir/position.h` `ExportPosition`); the
        // channel-enable mask rides in `ShaderInstruction::export_enable`.
        Pos1Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[POS1, S0, S1, S2, S3]),
        Pos2Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[POS2, S0, S1, S2, S3]),
        Pos3Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[POS3, S0, S1, S2, S3]),
        PrimVsrc0OffOffOffDone = format_define(&[PRIM, S0, OFF, OFF, OFF, DONE]),
        /// `exp null off,off,off,off [done] [vm]` — EXP target 9 with no
        /// channels enabled. Exports nothing; see [`NULL_TGT`].
        NullOffOffOffOffVmDone = format_define(&[NULL_TGT, OFF, OFF, OFF, OFF, VM, DONE]),
        Saddr = format_define(&[S0A2]),
        /// Beyond Kyty (SharpEmu PR #587): FLAT-class memory operand shape —
        /// `dst` (load dest / store data), `src[0]` VGPR address (a 64-bit pair
        /// when flat-addressed, else a 32-bit offset), `src[1]` SGPR base pair
        /// (NULL when flat-addressed), `src[2]` immediate byte offset.
        FlatAddr = format_define(&[D, S0A2, S1A2, S2]),
        Sdst2 = format_define(&[DA2]),
        SdstSbaseSoffset = format_define(&[D, S0A2, S1]),
        Sdst16SvSoffset = format_define(&[DA16, S0A4, S1]),
        Sdst2Ssrc02 = format_define(&[DA2, S0A2]),
        Sdst2Ssrc02Ssrc1 = format_define(&[DA2, S0A2, S1]),
        Sdst2Ssrc02Ssrc12 = format_define(&[DA2, S0A2, S1A2]),
        Sdst2SvSoffset = format_define(&[DA2, S0A4, S1]),
        Sdst4SbaseSoffset = format_define(&[DA4, S0A2, S1]),
        Sdst4SvSoffset = format_define(&[DA4, S0A4, S1]),
        Sdst8SbaseSoffset = format_define(&[DA8, S0A2, S1]),
        /// Beyond Kyty (upstream `KYTY_NI`s SMEM/SMRD opcode 0x04):
        /// `s_load_dwordx16 s[dst:dst+15], s[base:base+1], soffset` — 64 bytes
        /// into 16 consecutive SGPRs. Same operand shape as the x2/x4/x8 rows,
        /// only wider. Measured on Avatar: Frontiers of Pandora.
        Sdst16SbaseSoffset = format_define(&[DA16, S0A2, S1]),
        Sdst8SvSoffset = format_define(&[DA8, S0A4, S1]),
        SdstSvSoffset = format_define(&[D, S0A4, S1]),
        SmaskVsrc0Vsrc1 = format_define(&[DA2, S0, S1]),
        Ssrc0Ssrc1 = format_define(&[S0, S1]),
        SVdstSVsrc0 = format_define(&[D, S0]),
        SVdstSVsrc0SVsrc1 = format_define(&[D, S0, S1]),
        Vdata1Vaddr3StDmask1 = format_define(&[D, S0A3, S1A8, DMASK_1]),
        Vdata1Vaddr3StSsDmask1 = format_define(&[D, S0A3, S1A8, S2A4, DMASK_1]),
        /// Beyond Kyty: single-channel sample selecting .y (dmask 0x2) —
        /// measured on ASTRO.BOT `image_sample_lz` (MIMG 0x27 dmask 0x2).
        Vdata1Vaddr3StSsDmask2 = format_define(&[D, S0A3, S1A8, S2A4, DMASK_2]),
        Vdata1Vaddr3StSsDmask8 = format_define(&[D, S0A3, S1A8, S2A4, DMASK_8]),
        /// Beyond Kyty: one-channel `image_sample_lz_o` with packed offset
        /// plus 2D coordinates. Measured on ASTRO.BOT PS MIMG 0x37 dmask 0x1.
        Vdata1Vaddr4StSsDmask1 = format_define(&[D, S0A4, S1A8, S2A4, DMASK_1]),
        /// Same offset sample selecting channel Y; exposed after the dmask-1
        /// instruction in the same measured ASTRO.BOT pixel shader.
        Vdata1Vaddr4StSsDmask2 = format_define(&[D, S0A4, S1A8, S2A4, DMASK_2]),
        Vdata1VaddrSvSoffsIdxen = format_define(&[D, S0, S1A4, S2, IDXEN]),
        // Beyond Kyty: MUBUF single-dword addressing variants for idxen==0
        // and/or offen==1 (upstream EXIT_NOT_IMPLEMENTEDs both flags,
        // ShaderParse.cpp L2569-2570). Same model as the Vdata4* quartet the
        // BufferLoadDwordX4 rows already use.
        Vdata1SvSoffs = format_define(&[D, S1A4, S2]),
        Vdata1VaddrSvSoffsOffen = format_define(&[D, S0, S1A4, S2, OFFEN]),
        Vdata1Vaddr2SvSoffsOffenIdxen = format_define(&[D, S0A2, S1A4, S2, OFFEN, IDXEN]),
        Vdata1VaddrSvSoffsIdxenFloat1 = format_define(&[D, S0, S1A4, S2, IDXEN, FLOAT1]),
        Vdata2Vaddr3StSsDmask3 = format_define(&[DA2, S0A3, S1A8, S2A4, DMASK_3]),
        Vdata2Vaddr3StSsDmask5 = format_define(&[DA2, S0A3, S1A8, S2A4, DMASK_5]),
        Vdata2Vaddr3StSsDmask9 = format_define(&[DA2, S0A3, S1A8, S2A4, DMASK_9]),
        /// Beyond Kyty: two-channel unsampled fetch — measured on ASTRO.BOT
        /// `image_load` (MIMG 0x00 dmask 0x3).
        Vdata2Vaddr3StDmask3 = format_define(&[DA2, S0A3, S1A8, DMASK_3]),
        /// Beyond Kyty: two-channel ZW unsampled fetch (dmask 0xc).
        Vdata2Vaddr3StDmaskC = format_define(&[DA2, S0A3, S1A8, DMASK_C]),
        Vdata2VaddrStDmask3 = format_define(&[DA2, S0, S1A8, DMASK_3]),
        Vdata2VaddrSvSoffsIdxen = format_define(&[DA2, S0, S1A4, S2, IDXEN]),
        // Beyond Kyty: the two-dword MUBUF addressing variants completing the
        // flexible quartet for `buffer_load_dwordx2` (measured on ASTRO.BOT
        // scene compute) — same model as the Vdata1/Vdata4 sets.
        Vdata2SvSoffs = format_define(&[DA2, S1A4, S2]),
        Vdata2VaddrSvSoffsOffen = format_define(&[DA2, S0, S1A4, S2, OFFEN]),
        Vdata2Vaddr2SvSoffsOffenIdxen = format_define(&[DA2, S0A2, S1A4, S2, OFFEN, IDXEN]),
        Vdata3Vaddr3StDmask7 = format_define(&[DA3, S0A3, S1A8, DMASK_7]),
        Vdata3Vaddr3StSsDmask7 = format_define(&[DA3, S0A3, S1A8, S2A4, DMASK_7]),
        Vdata3Vaddr4StSsDmask7 = format_define(&[DA3, S0A4, S1A8, S2A4, DMASK_7]),
        Vdata3VaddrSvSoffsIdxen = format_define(&[DA3, S0, S1A4, S2, IDXEN]),
        // Beyond Kyty: the three-dword MUBUF addressing variants completing
        // the flexible quartet for `buffer_load_dwordx3` (measured on
        // ASTRO.BOT scene compute) — same model as the Vdata1/2/4 sets.
        Vdata3SvSoffs = format_define(&[DA3, S1A4, S2]),
        Vdata3VaddrSvSoffsOffen = format_define(&[DA3, S0, S1A4, S2, OFFEN]),
        Vdata3Vaddr2SvSoffsOffenIdxen = format_define(&[DA3, S0A2, S1A4, S2, OFFEN, IDXEN]),
        Vdata4Vaddr2SvSoffsOffenIdxen = format_define(&[DA4, S0A2, S1A4, S2, OFFEN, IDXEN]),
        Vdata4Vaddr2SvSoffsOffenIdxenFloat4 =
            format_define(&[DA4, S0A2, S1A4, S2, OFFEN, IDXEN, FLOAT4]),
        Vdata4Vaddr3StDmaskF = format_define(&[DA4, S0A3, S1A8, DMASK_F]),
        /// Beyond Kyty: four-texel single-channel gather — measured on
        /// ASTRO.BOT `image_gather4_lz` (MIMG 0x47 dmask 0x1): vdata is 4
        /// consecutive VGPRs (one per gathered texel) while the dmask names
        /// the one channel gathered.
        Vdata4Vaddr3StSsDmask1 = format_define(&[DA4, S0A3, S1A8, S2A4, DMASK_1]),
        /// Beyond Kyty: the same four-texel gather selecting channel Y —
        /// measured on ASTRO.BOT (`image_gather4_lz` dmask 0x2, the first
        /// blocker after the mixed-dim storage-image refusal was fixed).
        Vdata4Vaddr3StSsDmask2 = format_define(&[DA4, S0A3, S1A8, S2A4, DMASK_2]),
        /// Beyond Kyty: four-texel gather of channel Z. Same mechanism as the
        /// dmask 0x1/0x2 rows (the gather's SPIR-V `Component` operand is the
        /// dmask bit index), so it is decided by the encoding, not guessed.
        Vdata4Vaddr3StSsDmask4 = format_define(&[DA4, S0A3, S1A8, S2A4, DMASK_4]),
        /// Beyond Kyty: four-texel gather of channel W (see the dmask 0x4 row).
        Vdata4Vaddr3StSsDmask8 = format_define(&[DA4, S0A3, S1A8, S2A4, DMASK_8]),
        Vdata4Vaddr3StSsDmaskF = format_define(&[DA4, S0A3, S1A8, S2A4, DMASK_F]),
        Vdata4Vaddr4StDmaskF = format_define(&[DA4, S0A4, S1A8, DMASK_F]),
        Vdata4VaddrSvSoffsIdxen = format_define(&[DA4, S0, S1A4, S2, IDXEN]),
        Vdata4VaddrSvSoffsIdxenFloat4 = format_define(&[DA4, S0, S1A4, S2, IDXEN, FLOAT4]),
        Vdata4SvSoffs = format_define(&[DA4, S1A4, S2]),
        Vdata4VaddrSvSoffsOffen = format_define(&[DA4, S0, S1A4, S2, OFFEN]),
        VdstGds = format_define(&[D, GDS]),
        /// Beyond Kyty: `ds_read2_b32 vdst[2], addr [offset0] [offset1]` —
        /// dst = 2 consecutive VGPRs, src0 = address VGPR, src1/src2 = the
        /// two offsets as literal constants (stored in BYTES, i.e. the
        /// encoded dword-unit fields scaled by 4, so every DS recompiler
        /// indexes `%lds` the same way).
        Vdst2Vsrc0Vsrc1Vsrc2 = format_define(&[DA2, S0, S1, S2]),
        /// Beyond Kyty: `ds_read_b128 vdst[4], addr [offset]` — dst = 4
        /// consecutive VGPRs, src0 = address VGPR, src1 = the 16-bit byte
        /// offset as a literal constant (dword k reads at `offset + 4k`).
        Vdst4Vsrc0Vsrc1 = format_define(&[DA4, S0, S1]),
        /// Beyond Kyty: `ds_read_b96 vdst[3], addr [offset]` — the
        /// three-dword row of the same model (dword k reads at
        /// `offset + 4k`).
        Vdst3Vsrc0Vsrc1 = format_define(&[DA3, S0, S1]),
        /// Beyond Kyty: `ds_write_b32 addr, data0 [offset]` — src0 = address
        /// VGPR, src1 = data VGPR, src2 = the 16-bit instruction byte offset
        /// as a literal constant.
        Vsrc0Vsrc1Vsrc2 = format_define(&[S0, S1, S2]),
        /// Beyond Kyty: `ds_write_b96 addr, data0[3] [offset]` — src0 =
        /// address VGPR, src1 = first of 3 consecutive data VGPRs, src2 =
        /// the 16-bit instruction byte offset as a literal constant.
        Vsrc0Vsrc13Vsrc2 = format_define(&[S0, S1A3, S2]),
        /// Beyond Kyty: `ds_write_b128 addr, data0[4] [offset]` — the
        /// four-dword row of the same model.
        Vsrc0Vsrc14Vsrc2 = format_define(&[S0, S1A4, S2]),
        VdstSdst2Vsrc0Vsrc1 = format_define(&[D, D2A2, S0, S1]),
        /// Beyond Kyty: `v_add_co_ci_u32 vdst, sdst, src0, src1, ssrc2` — the
        /// add-with-carry shape. dst = VGPR sum, dst2 = carry-out mask (2
        /// dwords), src0/src1 = the addends, src2 = carry-in mask (2 dwords).
        /// The VOP2 form fills dst2 and src2 with VCC; the VOP3B form reads
        /// them from the sdst / ssrc2 fields.
        VdstSdst2Vsrc0Vsrc1Smask2 = format_define(&[D, D2A2, S0, S1, S2A2]),
        VdstVsrc0Vsrc1Smask2 = format_define(&[D, S0, S1, S2A2]),
        VdstVsrc0Vsrc1Vsrc2 = format_define(&[D, S0, S1, S2]),
        VdstVsrcAttrChan = format_define(&[D, S0, ATTR]),
    }
}

/// Kyty: Shader.h `ShaderInstructionTypeFormat` (L361).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderInstructionTypeFormat {
    pub type_: ShaderInstructionType,
    pub format: shader_instruction_format::Format,
}

/// Kyty: Shader.h `ShaderOperandType` (L367).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShaderOperandType {
    #[default]
    Unknown,
    LiteralConstant,
    IntegerInlineConstant,
    FloatInlineConstant,
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    ExecZ,
    Scc,
    Vgpr,
    Sgpr,
    M0,
    Null,
}

/// Kyty: Shader.h `ShaderConstant` union (L385). Rust stores the raw 32 bits
/// (`u`) and reinterprets on access instead of a C union.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct ShaderConstant {
    pub u: u32,
}

impl ShaderConstant {
    #[must_use]
    pub const fn from_u(u: u32) -> Self {
        Self { u }
    }

    #[must_use]
    pub const fn from_i(i: i32) -> Self {
        Self { u: i as u32 }
    }

    #[must_use]
    pub const fn from_f(f: f32) -> Self {
        Self { u: f.to_bits() }
    }

    /// The union's `.i` view.
    #[must_use]
    pub const fn i(self) -> i32 {
        self.u as i32
    }

    /// The union's `.f` view.
    #[must_use]
    pub const fn f(self) -> f32 {
        f32::from_bits(self.u)
    }
}

impl std::fmt::Debug for ShaderConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:08x}", self.u)
    }
}

/// Beyond Kyty: the cross-lane pattern of a DPP (Data-Parallel Primitives)
/// source. Two hardware sub-forms, distinguished by the VOP `src0` marker:
/// `0xfa` (DPP16) and `0xe9`/`0xea` (DPP8/DPP8FI). shadPS4's `struct Dpp` /
/// `DppCtrl` (GPL-2.0) were studied, not copied.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DppMode {
    /// DPP16: a 9-bit `dpp_ctrl` selects the cross-lane pattern within each row
    /// of 16 lanes (`quad_perm` 0x000-0x0ff, `row_shl` 0x101-0x10f, `row_shr`
    /// 0x111-0x11f, `row_ror` 0x121-0x12f, `row_mirror` 0x140, `row_half_mirror`
    /// 0x141). The raw control is kept; interpretation is the recompiler's job.
    Dpp16 { ctrl: u16 },
    /// DPP8: eight independent 3-bit lane selects, one per lane in each group of
    /// eight (`lane_sel[i]` is the source lane for output lane `i`).
    Dpp8 { lane_sel: [u8; 8] },
}

/// Beyond Kyty: decoded DPP control carried on a `ShaderOperand` (`src0` only —
/// DPP is legal only as src0). Row/bank masks and `bound_ctrl` apply to DPP16;
/// they stay zero/false for DPP8. `fetch_inactive` is DPP16's `fi` bit or the
/// DPP8FI marker (`0xea`). See [`DppMode`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DppCtrl {
    pub mode: DppMode,
    pub row_mask: u8,
    pub bank_mask: u8,
    pub bound_ctrl: bool,
    pub fetch_inactive: bool,
}

/// Packed (VOP3P) source and destination modifiers.
///
/// Beyond Kyty; ported from SharpEmu's `Gen5Vop3pControl`
/// (`Gen5ShaderIr.cs`, PR #460 `472fc96` for the clamp bit). Each mask holds
/// one bit per source operand, bit `i` for `src[i]`.
///
/// These cannot live on [`ShaderOperand`] the way VOP3A's `negate`/`absolute`
/// do: VOP3P carries TWO negate masks because the two 16-bit result lanes
/// negate their halves independently, and `op_sel`/`op_sel_hi` select a half
/// per lane. One `bool` per operand cannot express that.
///
/// For the three MIX ops (`V_FMA_MIX_*`) the same fields are reinterpreted:
/// `op_sel_hi` bit `i` means "read `src[i]` as an f16" (rather than as a full
/// f32), `op_sel` bit `i` picks which half, `neg_hi` bit `i` is the
/// ABSOLUTE-value modifier and `neg_lo` bit `i` negates — applied abs-then-neg.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Vop3pControl {
    /// `op_sel` (word0 bits [13:11]): which 16-bit half of each source feeds
    /// the LOW result lane.
    pub op_sel: u32,
    /// `op_sel_hi` (word1 bits [28:27] plus word0 bit 14 as bit 2): which half
    /// feeds the HIGH result lane.
    pub op_sel_hi: u32,
    /// `neg` / `neg_lo` (word1 bits [31:29]): negate the value routed to the
    /// low lane.
    pub neg_lo: u32,
    /// `neg_hi` (word0 bits [10:8]): negate the value routed to the high lane.
    pub neg_hi: u32,
    /// `clamp` (word0 bit 15): saturate each output half to `[0, 1]`.
    pub clamp: bool,
}

/// Kyty: Shader.h `ShaderOperand` (L392).
#[derive(Copy, Clone, Debug)]
pub struct ShaderOperand {
    pub type_: ShaderOperandType,
    pub constant: ShaderConstant,
    pub register_id: i32,
    pub size: i32,
    pub multiplier: f32,
    pub absolute: bool,
    pub negate: bool,
    pub clamp: bool,
    /// Beyond Kyty: SDWA sub-dword source select (`src{0,1}_sel`). 6 = DWORD
    /// (the whole register, the non-SDWA default); 0-3 = BYTE_0..BYTE_3;
    /// 4-5 = WORD_0..WORD_1. The operand loaders extract the selected lane
    /// (shift + mask, zero-extended — `sext` stays a named parse refusal)
    /// before the operation consumes it. Measured on ASTRO.BOT scene compute
    /// (vopc src1_sel and vop1 src0_sel).
    pub lane_sel: u8,
    /// Beyond Kyty: DPP (Data-Parallel Primitives) cross-lane control, present
    /// only on a DPP-form src0 (VOP1/VOP2/VOPC `src0 == 0xfa`/`0xe9`/`0xea`).
    /// `None` is the ordinary same-lane operand. The parser decodes it so the
    /// instruction is the correct two dwords and shader boundaries stay in
    /// sync; the recompiler refuses it by name (no wave-level model yet). See
    /// [`DppCtrl`].
    pub dpp: Option<DppCtrl>,
}

impl Default for ShaderOperand {
    fn default() -> Self {
        Self {
            type_: ShaderOperandType::Unknown,
            constant: ShaderConstant::default(),
            register_id: 0,
            size: 0,
            multiplier: 1.0,
            absolute: false,
            negate: false,
            clamp: false,
            lane_sel: 6,
            dpp: None,
        }
    }
}

/// Kyty equality (Shader.h L403) deliberately ignores the modifiers
/// (`multiplier`/`absolute`/`negate`/`clamp`).
impl PartialEq for ShaderOperand {
    fn eq(&self, other: &Self) -> bool {
        self.type_ == other.type_
            && self.constant.u == other.constant.u
            && self.register_id == other.register_id
            && self.size == other.size
    }
}

/// Kyty: Shader.h `ShaderInstruction` (L409).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderInstruction {
    pub pc: u32,
    pub type_: ShaderInstructionType,
    pub format: shader_instruction_format::Format,
    pub src: [ShaderOperand; 4],
    pub src_num: i32,
    pub dst: ShaderOperand,
    pub dst2: ShaderOperand,
    /// EXP channel-enable mask (`en`): which of the four `vsrc` channels this
    /// export actually writes. Only meaningful for `type_ == Exp`; a full
    /// export is `0xf`, a partial one (e.g. a `vec2` texcoord) `0x3`. Set by
    /// `shader_parse_exp`; the recompiler writes 0 to the disabled channels.
    pub export_enable: u32,
    /// Raw EXP target. Fragment colour targets 0..=7 map directly to Vulkan
    /// output locations 0..=7. Kept separately from `format` so the existing
    /// MRT0 operand-shape rows can lower every MRT without multiplying the
    /// format enum by eight identical variants.
    pub export_target: u8,
    /// Beyond Kyty (SharpEmu PR #587 `Gen5GlobalMemoryControl.UsesFlatAddress`):
    /// only meaningful for the FLAT-class ops (`Flat*`). `true` when the guest
    /// address is a complete 64-bit pointer held in the VGPR pair
    /// `(src[0], src[0]+1)` — the FLAT segment, or a GLOBAL segment whose SADDR
    /// was NULL. `false` when a GLOBAL op supplies an SGPR base pair (`src[1]`)
    /// and `src[0]` is a 32-bit per-lane offset. Gates the address computation
    /// in the recompiler exactly as SharpEmu gates its SPIR-V emission.
    pub uses_flat_address: bool,
    /// GFX10 MIMG non-sequential-address (NSA) payload. Address component zero
    /// always comes from `src[0]`; when this count is non-zero, subsequent
    /// components come from these explicitly encoded VGPRs instead of
    /// consecutive registers. One dword carries four byte-sized VGPR ids.
    pub mimg_nsa_dwords: u8,
    pub mimg_nsa_addr: [ShaderOperand; 12],
    /// Beyond Kyty: packed-math modifiers, `Some` only for the VOP3P
    /// instructions (`VPk*`, `VFmaMix*`). See [`Vop3pControl`].
    pub vop3p: Option<Vop3pControl>,
}

/// Kyty: Shader.h `ShaderLabel` (L420). `dst = pc + 4 + src[0].constant.i`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShaderLabel {
    dst: u32,
    src: u32,
}

impl ShaderLabel {
    #[must_use]
    pub const fn new(dst: u32, src: u32) -> Self {
        Self { dst, src }
    }

    /// Kyty: `ShaderLabel(const ShaderInstruction&)` (Shader.h L424).
    #[must_use]
    pub fn from_instruction(inst: &ShaderInstruction) -> Self {
        Self {
            dst: inst
                .pc
                .wrapping_add(4)
                .wrapping_add(inst.src[0].constant.i() as u32),
            src: inst.pc,
        }
    }

    #[must_use]
    pub const fn get_dst(&self) -> u32 {
        self.dst
    }

    #[must_use]
    pub const fn get_src(&self) -> u32 {
        self.src
    }

    pub fn disable(&mut self) {
        self.dst = 0;
        self.src = 0;
    }

    #[must_use]
    pub const fn is_disabled(&self) -> bool {
        self.dst == 0 && self.src == 0
    }
}

/// Kyty: `ShaderLabel::ToString()` (Shader.h L431) — exposed as `Display`
/// (and thereby `.to_string()`) in Rust.
impl std::fmt::Display for ShaderLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "label_{:04x}_{:04x}", self.dst, self.src)
    }
}

/// Kyty: Shader.h `ShaderDebugPrintf::Type` (L448).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShaderDebugPrintfType {
    Uint,
    Int,
    Float,
}

/// Kyty: Shader.h `ShaderDebugPrintf` (L446) — a debug-printf command
/// injected at `pc`. The data model is ported; the global injection registry
/// (`g_debug_printfs`, Shader.cpp L100/L3006) is not — see `analysis.rs`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderDebugPrintf {
    pub pc: u32,
    pub format: String,
    pub types: Vec<ShaderDebugPrintfType>,
    pub args: Vec<ShaderOperand>,
}

/// Kyty: Shader.h `ShaderControlFlowBlock` (L460).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderControlFlowBlock {
    pub pc: u32,
    pub is_discard: bool,
    pub is_valid: bool,
    pub last: ShaderInstruction,
}

/// Kyty: Shader.cpp `operand_to_str` (L117). Kyty EXITs on inconsistent
/// sizes/modifiers; the port renders what it can instead.
fn operand_to_str(op: &ShaderOperand) -> String {
    use ShaderOperandType as O;
    match op.type_ {
        O::LiteralConstant => return format!("{:.6} ({})", op.constant.f(), op.constant.u),
        O::IntegerInlineConstant => return format!("{}", op.constant.i()),
        O::FloatInlineConstant => return format!("{:.6}", op.constant.f()),
        _ => {}
    }

    let mut ret = match op.type_ {
        O::VccHi => "vcc_hi".to_string(),
        O::VccLo => "vcc_lo".to_string(),
        O::ExecHi => "exec_hi".to_string(),
        O::ExecLo => "exec_lo".to_string(),
        O::ExecZ => "execz".to_string(),
        O::Scc => "scc".to_string(),
        O::M0 => "m0".to_string(),
        O::Vgpr => format!("v{}", op.register_id),
        O::Sgpr => format!("s{}", op.register_id),
        O::Null => "null".to_string(),
        _ => "???".to_string(),
    };

    if op.absolute {
        ret = format!("abs({ret})");
    }
    if op.negate {
        return format!("-{ret}");
    }
    ret
}

/// Kyty: Shader.cpp `operand_array_to_str` (L170).
fn operand_array_to_str(op: &ShaderOperand, n: i32) -> String {
    use ShaderOperandType as O;
    let mut ret = match op.type_ {
        O::VccLo if n == 2 => "vcc".to_string(),
        O::ExecLo if n == 2 => "exec".to_string(),
        O::Sgpr => format!("s[{}:{}]", op.register_id, op.register_id + n - 1),
        O::Vgpr => format!("v[{}:{}]", op.register_id, op.register_id + n - 1),
        O::LiteralConstant if n == 2 => format!("{:.6} ({})", op.constant.f(), op.constant.u),
        O::IntegerInlineConstant if n == 2 => format!("{}", op.constant.i()),
        _ => "???".to_string(),
    };

    if op.absolute {
        ret = format!("abs({ret})");
    }
    if op.negate {
        return format!("-{ret}");
    }
    ret
}

/// Kyty: Shader.cpp `dbg_fmt_print` (L282). Walks the packed Format bytes,
/// low byte first, prepending — so the rendered operand order matches the
/// token order in `FormatDefine`.
fn dbg_fmt_print(inst: &ShaderInstruction) -> String {
    use shader_instruction_format as sif;
    use shader_instruction_format::Format;

    let mut f = inst.format as u64;
    if inst.format == Format::Unknown || inst.format == Format::Empty {
        return String::new();
    }
    let mut str = String::new();
    loop {
        let fu = f & 0xff;
        if fu == 0 {
            break;
        }
        let s = match fu {
            sif::D => operand_to_str(&inst.dst),
            sif::D2 => operand_to_str(&inst.dst2),
            sif::S0 => operand_to_str(&inst.src[0]),
            sif::S1 => operand_to_str(&inst.src[1]),
            sif::S2 => operand_to_str(&inst.src[2]),
            sif::S3 => operand_to_str(&inst.src[3]),
            sif::DA2 => operand_array_to_str(&inst.dst, 2),
            sif::DA3 => operand_array_to_str(&inst.dst, 3),
            sif::DA4 => operand_array_to_str(&inst.dst, 4),
            sif::DA8 => operand_array_to_str(&inst.dst, 8),
            sif::DA16 => operand_array_to_str(&inst.dst, 16),
            sif::D2A2 => operand_array_to_str(&inst.dst2, 2),
            sif::D2A3 => operand_array_to_str(&inst.dst2, 3),
            sif::D2A4 => operand_array_to_str(&inst.dst2, 4),
            sif::S0A2 => operand_array_to_str(&inst.src[0], 2),
            sif::S0A3 => operand_array_to_str(&inst.src[0], 3),
            sif::S0A4 => operand_array_to_str(&inst.src[0], 4),
            sif::S1A2 => operand_array_to_str(&inst.src[1], 2),
            sif::S1A3 => operand_array_to_str(&inst.src[1], 3),
            sif::S1A4 => operand_array_to_str(&inst.src[1], 4),
            sif::S1A8 => operand_array_to_str(&inst.src[1], 8),
            sif::S2A2 => operand_array_to_str(&inst.src[2], 2),
            sif::S2A3 => operand_array_to_str(&inst.src[2], 3),
            sif::S2A4 => operand_array_to_str(&inst.src[2], 4),
            sif::ATTR => format!("attr{}.{}", inst.src[1].constant.u, inst.src[2].constant.u),
            sif::IDXEN => "idxen".to_string(),
            sif::OFFEN => "offen".to_string(),
            sif::FLOAT1 => "format:float1".to_string(),
            sif::FLOAT4 => "format:float4".to_string(),
            sif::POS0 => "pos0".to_string(),
            sif::DONE => "done".to_string(),
            sif::PARAM0 => "param0".to_string(),
            sif::PARAM1 => "param1".to_string(),
            sif::PARAM2 => "param2".to_string(),
            sif::PARAM3 => "param3".to_string(),
            sif::PARAM4 => "param4".to_string(),
            sif::MRT0 => "mrt_color0".to_string(),
            sif::PRIM => "prim".to_string(),
            sif::OFF => "off".to_string(),
            sif::COMPR => "compr".to_string(),
            sif::VM => "vm".to_string(),
            sif::L => format!(
                "label_{:04x}",
                inst.pc
                    .wrapping_add(4)
                    .wrapping_add(inst.src[0].constant.i() as u32)
            ),
            sif::POS1 => "pos1".to_string(),
            sif::POS2 => "pos2".to_string(),
            sif::POS3 => "pos3".to_string(),
            sif::DMASK_1 => "dmask:0x1".to_string(),
            sif::DMASK_2 => "dmask:0x2".to_string(),
            sif::DMASK_8 => "dmask:0x8".to_string(),
            sif::DMASK_3 => "dmask:0x3".to_string(),
            sif::DMASK_5 => "dmask:0x5".to_string(),
            sif::DMASK_7 => "dmask:0x7".to_string(),
            sif::DMASK_9 => "dmask:0x9".to_string(),
            sif::DMASK_4 => "dmask:0x4".to_string(),
            sif::DMASK_C => "dmask:0xc".to_string(),
            sif::DMASK_F => "dmask:0xf".to_string(),
            sif::GDS => "gds".to_string(),
            _ => "???".to_string(),
        };
        str = if str.is_empty() {
            s
        } else {
            format!("{s}, {str}")
        };
        f >>= 8;
    }
    if inst.dst.multiplier == 2.0 {
        str += " mul:2";
    }
    if inst.dst.multiplier == 4.0 {
        str += " mul:4";
    }
    if inst.dst.multiplier == 0.5 {
        str += " div:2";
    }
    if inst.dst.clamp {
        str += " clamp";
    }
    str
}

/// Kyty: Shader.cpp `IsDiscardInstruction` (L428).
fn is_discard_instruction(code: &[ShaderInstruction], index: usize) -> bool {
    use ShaderInstructionType as T;
    use shader_instruction_format::Format;
    if index == 0 || index + 1 >= code.len() {
        return false;
    }
    let prev_inst = &code[index - 1];
    let inst = &code[index];
    let next_inst = &code[index + 1];

    inst.type_ == T::Exp
        && inst.format == Format::Mrt0OffOffComprVmDone
        && prev_inst.type_ == T::SMovB64
        && prev_inst.format == Format::Sdst2Ssrc02
        && prev_inst.dst.type_ == ShaderOperandType::ExecLo
        && prev_inst.src[0].type_ == ShaderOperandType::IntegerInlineConstant
        && prev_inst.src[0].constant.i() == 0
        && next_inst.type_ == T::SEndpgm
}

/// Kyty: Shader.h `ShaderCode` (L468).
#[derive(Clone, Debug)]
pub struct ShaderCode {
    /// Absolute guest address of instruction dword zero. Kyty keeps this in the
    /// raw code pointer; the bounds-checked Rust port records it explicitly so
    /// PC-relative scalar instructions retain their real 64-bit value.
    base_address: u64,
    hash0: u32,
    crc32: u32,
    instructions: Vec<ShaderInstruction>,
    labels: Vec<ShaderLabel>,
    indirect_labels: Vec<ShaderLabel>,
    type_: ShaderType,
    debug_printfs: Vec<ShaderDebugPrintf>,
    vs_embedded_id: u32,
    ps_embedded_id: u32,
    vs_embedded: bool,
    ps_embedded: bool,
}

impl Default for ShaderCode {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderCode {
    /// Kyty ctor pre-expands the instruction vector to 128 entries.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_address: 0,
            hash0: 0,
            crc32: 0,
            instructions: Vec::with_capacity(128),
            labels: Vec::new(),
            indirect_labels: Vec::new(),
            type_: ShaderType::Unknown,
            debug_printfs: Vec::new(),
            vs_embedded_id: 0,
            ps_embedded_id: 0,
            vs_embedded: false,
            ps_embedded: false,
        }
    }

    #[must_use]
    pub fn get_instructions(&self) -> &Vec<ShaderInstruction> {
        &self.instructions
    }

    pub fn get_instructions_mut(&mut self) -> &mut Vec<ShaderInstruction> {
        &mut self.instructions
    }

    #[must_use]
    pub const fn get_base_address(&self) -> u64 {
        self.base_address
    }

    pub fn set_base_address(&mut self, address: u64) {
        self.base_address = address;
    }

    #[must_use]
    pub fn get_labels(&self) -> &Vec<ShaderLabel> {
        &self.labels
    }

    pub fn get_labels_mut(&mut self) -> &mut Vec<ShaderLabel> {
        &mut self.labels
    }

    #[must_use]
    pub fn get_indirect_labels(&self) -> &Vec<ShaderLabel> {
        &self.indirect_labels
    }

    pub fn get_indirect_labels_mut(&mut self) -> &mut Vec<ShaderLabel> {
        &mut self.indirect_labels
    }

    #[must_use]
    pub const fn get_type(&self) -> ShaderType {
        self.type_
    }

    pub fn set_type(&mut self, type_: ShaderType) {
        self.type_ = type_;
    }

    /// Kyty: `GetDebugPrintfs` (Shader.h L487).
    #[must_use]
    pub fn get_debug_printfs(&self) -> &Vec<ShaderDebugPrintf> {
        &self.debug_printfs
    }

    pub fn get_debug_printfs_mut(&mut self) -> &mut Vec<ShaderDebugPrintf> {
        &mut self.debug_printfs
    }

    /// Kyty: `HasAnyOf` (Shader.h L491).
    #[must_use]
    pub fn has_any_of(&self, types: &[ShaderInstructionType]) -> bool {
        types
            .iter()
            .any(|t| self.instructions.iter().any(|inst| inst.type_ == *t))
    }

    #[must_use]
    pub const fn is_vs_embedded(&self) -> bool {
        self.vs_embedded
    }

    pub fn set_vs_embedded(&mut self, embedded: bool) {
        self.vs_embedded = embedded;
    }

    #[must_use]
    pub const fn get_vs_embedded_id(&self) -> u32 {
        self.vs_embedded_id
    }

    pub fn set_vs_embedded_id(&mut self, embedded_id: u32) {
        self.vs_embedded_id = embedded_id;
    }

    #[must_use]
    pub const fn is_ps_embedded(&self) -> bool {
        self.ps_embedded
    }

    pub fn set_ps_embedded(&mut self, embedded: bool) {
        self.ps_embedded = embedded;
    }

    #[must_use]
    pub const fn get_ps_embedded_id(&self) -> u32 {
        self.ps_embedded_id
    }

    pub fn set_ps_embedded_id(&mut self, embedded_id: u32) {
        self.ps_embedded_id = embedded_id;
    }

    #[must_use]
    pub const fn get_crc32(&self) -> u32 {
        self.crc32
    }

    pub fn set_crc32(&mut self, c: u32) {
        self.crc32 = c;
    }

    #[must_use]
    pub const fn get_hash0(&self) -> u32 {
        self.hash0
    }

    pub fn set_hash0(&mut self, h: u32) {
        self.hash0 = h;
    }

    /// Kyty: Shader.cpp `DbgInstructionToStr` (L397).
    #[must_use]
    pub fn dbg_instruction_to_str(inst: &ShaderInstruction) -> String {
        let name = format!("{:?}", inst.type_);
        let format = format!("{:?}", inst.format);
        format!("{name:<20} [{format:<30}] {}", dbg_fmt_print(inst))
    }

    /// Kyty: Shader.cpp `DbgDump` (L410).
    #[must_use]
    pub fn dbg_dump(&self) -> String {
        let mut ret = String::new();
        for inst in &self.instructions {
            if self
                .labels
                .iter()
                .any(|label| !label.is_disabled() && label.get_dst() == inst.pc)
            {
                let _ = write!(ret, "\nlabel_{:04x}:\n", inst.pc);
            }
            if self
                .indirect_labels
                .iter()
                .any(|label| !label.is_disabled() && label.get_dst() == inst.pc)
            {
                ret.push('\n');
            }
            let _ = writeln!(ret, "  {}", Self::dbg_instruction_to_str(inst));
        }
        ret
    }

    /// Kyty: Shader.cpp `ReadBlock` (L474).
    #[must_use]
    pub fn read_block(&self, pc: u32) -> ShaderControlFlowBlock {
        use ShaderInstructionType as T;
        let mut ret = ShaderControlFlowBlock::default();
        if let Some(index) = self.instructions.iter().position(|inst| inst.pc == pc) {
            ret.pc = pc;
            ret.is_valid = true;
            for i in index..self.instructions.len() {
                let inst = &self.instructions[i];
                if matches!(
                    inst.type_,
                    T::SEndpgm
                        | T::SCbranchExecz
                        | T::SCbranchScc0
                        | T::SCbranchScc1
                        | T::SCbranchVccz
                        | T::SCbranchVccnz
                        | T::SBranch
                ) {
                    ret.last = *inst;
                    break;
                }
                if is_discard_instruction(&self.instructions, i) {
                    ret.is_discard = true;
                }
            }
        }
        ret
    }

    /// Kyty: Shader.cpp `ReadIntructions` (L509) — Kyty's spelling (sic).
    #[must_use]
    pub fn read_intructions(&self, block: &ShaderControlFlowBlock) -> Vec<ShaderInstruction> {
        let mut ret = Vec::new();
        if let Some(index) = self
            .instructions
            .iter()
            .position(|inst| inst.pc == block.pc)
        {
            for inst in &self.instructions[index..] {
                ret.push(*inst);
                if inst.pc == block.last.pc {
                    break;
                }
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::shader_instruction_format::{Format, format_define};
    use super::*;

    #[test]
    fn format_define_packs_bytes_first_token_highest() {
        // Kyty: Shader.h FormatDefine (L293).
        use super::shader_instruction_format as sif;
        assert_eq!(format_define(&[sif::U]), 0);
        assert_eq!(format_define(&[sif::N]), 1);
        assert_eq!(format_define(&[sif::D, sif::S0]), (sif::D << 8) | sif::S0);
        assert_eq!(Format::SVdstSVsrc0 as u64, 0x0204);
        assert_eq!(Format::Label as u64, sif::L);
        assert_eq!(
            Format::Mrt0OffOffComprVmDone as u64,
            (sif::MRT0 << 40)
                | (sif::OFF << 32)
                | (sif::OFF << 24)
                | (sif::COMPR << 16)
                | (sif::VM << 8)
                | sif::DONE
        );
    }

    #[test]
    fn shader_operand_equality_ignores_modifiers() {
        // Kyty: Shader.h ShaderOperand::operator== (L403).
        let a = ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: 3,
            size: 1,
            ..Default::default()
        };
        let mut b = a;
        b.negate = true;
        b.absolute = true;
        b.multiplier = 4.0;
        b.clamp = true;
        assert_eq!(a, b);
        b.register_id = 4;
        assert_ne!(a, b);
    }

    #[test]
    fn shader_label_from_instruction() {
        // Kyty: Shader.h ShaderLabel(const ShaderInstruction&) (L424).
        let mut inst = ShaderInstruction {
            pc: 8,
            ..Default::default()
        };
        inst.src[0].type_ = ShaderOperandType::LiteralConstant;
        inst.src[0].constant = ShaderConstant::from_i(-12);
        let label = ShaderLabel::from_instruction(&inst);
        assert_eq!(label.get_dst(), 0);
        assert_eq!(label.get_src(), 8);
        assert_eq!(label.to_string(), "label_0000_0008");
        // IsDisabled needs dst == 0 AND src == 0 (Shader.h L439); src is 8.
        assert!(!label.is_disabled());
    }

    #[test]
    fn shader_label_disable() {
        let mut label = ShaderLabel::new(0x10, 0x4);
        assert!(!label.is_disabled());
        label.disable();
        assert!(label.is_disabled());
    }

    #[test]
    fn dbg_instruction_to_str_s_mov() {
        let mut inst = ShaderInstruction {
            type_: ShaderInstructionType::SMovB32,
            format: Format::SVdstSVsrc0,
            src_num: 1,
            ..Default::default()
        };
        inst.dst.type_ = ShaderOperandType::Sgpr;
        inst.dst.register_id = 0;
        inst.dst.size = 1;
        inst.src[0].type_ = ShaderOperandType::Sgpr;
        inst.src[0].register_id = 1;
        inst.src[0].size = 1;
        let s = ShaderCode::dbg_instruction_to_str(&inst);
        assert!(s.contains("SMovB32"), "{s}");
        assert!(s.contains("[SVdstSVsrc0"), "{s}");
        assert!(s.ends_with("s0, s1"), "{s}");
    }

    #[test]
    fn dbg_operand_modifiers() {
        let mut inst = ShaderInstruction {
            type_: ShaderInstructionType::VAddF32,
            format: Format::SVdstSVsrc0SVsrc1,
            src_num: 2,
            ..Default::default()
        };
        inst.dst.type_ = ShaderOperandType::Vgpr;
        inst.dst.size = 1;
        inst.dst.multiplier = 2.0;
        inst.dst.clamp = true;
        inst.src[0].type_ = ShaderOperandType::Vgpr;
        inst.src[0].register_id = 1;
        inst.src[0].size = 1;
        inst.src[0].absolute = true;
        inst.src[1].type_ = ShaderOperandType::Vgpr;
        inst.src[1].register_id = 2;
        inst.src[1].size = 1;
        inst.src[1].negate = true;
        let s = ShaderCode::dbg_instruction_to_str(&inst);
        assert!(s.contains("v0, abs(v1), -v2"), "{s}");
        assert!(s.contains(" mul:2"), "{s}");
        assert!(s.contains(" clamp"), "{s}");
    }
}
