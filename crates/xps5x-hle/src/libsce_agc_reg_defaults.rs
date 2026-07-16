//! AGC Gen5 register-default tables served by `sceAgcGetRegisterDefaults2`
//! and `sceAgcGetRegisterDefaults2Internal`.
//!
//! Ported from Kyty Graphics.cpp (MIT (c) InoriRus): the `Gen5` tables
//! `g_cx_reg_info1/2`, `g_sh_reg_info1/2` and `g_uc_reg_info1/2`, with each
//! `Pm4::<NAME>` register offset resolved to its numeric value from Kyty's
//! `Pm4.h`. Every entry keeps Kyty's exact type hash, register offsets and
//! default values, in the original order (the index tables `g_tbl_index1/2`
//! are derived from these at materialization time, exactly as Kyty derives
//! them with its `KYTY_INDEX_*` macros).
//!
//! This file is mechanically generated from the Kyty sources; edit the
//! generator (see the port commit) rather than hand-tuning entries.

/// One register-defaults entry: `(type_hash, &[(register_offset, value)])`.
///
/// Kyty's `RegisterDefaultInfo` is `{ uint32_t type; ShaderRegister reg[16] }`
/// with unused trailing slots zeroed; the guest-memory materializer pads each
/// entry back out to 16 `(offset, value)` slots.
pub(crate) type RegisterDefaultInfo = (u32, &'static [(u32, u32)]);

#[rustfmt::skip]
pub(crate) static CX_REG_INFO1: &[RegisterDefaultInfo] = &[
    // CB_COLOR_CONTROL
    (0xe24f806d, &[(0x0202, 0x00cc0010)]),
    // CB_DCC_CONTROL
    (0xf6c28182, &[(0x0109, 0x00000000)]),
    // CB_RMI_GL2_CACHE_CONTROL
    (0x6f6e55a5, &[(0x0104, 0x00000000)]),
    // CB_SHADER_MASK
    (0x0bc65da4, &[(0x008f, 0x00000000)]),
    // CB_TARGET_MASK
    (0x9e5ad592, &[(0x008e, 0x00000000)]),
    // DB_ALPHA_TO_MASK
    (0xbb513b98, &[(0x02dc, 0x0000aa00)]),
    // DB_COUNT_CONTROL
    (0xab64b23b, &[(0x0001, 0x00000000)]),
    // DB_DEPTH_CONTROL
    (0x53c39964, &[(0x0200, 0x00000000)]),
    // DB_EQAA
    (0x01396b11, &[(0x0201, 0x00000000)]),
    // DB_RENDER_CONTROL
    (0x7d42019a, &[(0x0000, 0x00000000)]),
    // PS_SHADER_SAMPLE_EXCLUSION_MASK
    (0x3548f523, &[(0x0006, 0x00000000)]),
    // DB_RMI_L2_CACHE_CONTROL
    (0xf43ad28a, &[(0x001f, 0x00000000)]),
    // DB_SHADER_CONTROL
    (0x6de4c312, &[(0x0203, 0x00000000)]),
    // DB_SRESULTS_COMPARE_STATE0
    (0x00a77ae0, &[(0x02b0, 0x00000000)]),
    // DB_SRESULTS_COMPARE_STATE1
    (0x00a779b7, &[(0x02b1, 0x00000000)]),
    // DB_STENCILREFMASK
    (0x5100100c, &[(0x010c, 0x00000000)]),
    // DB_STENCILREFMASK_BF
    (0x59958bba, &[(0x010d, 0x00000000)]),
    // DB_STENCIL_CONTROL
    (0x0c06f17c, &[(0x010b, 0x00000000)]),
    // GE_MAX_OUTPUT_PER_SUBGROUP
    (0x6f104b72, &[(0x01ff, 0x00000000)]),
    // PA_CL_CLIP_CNTL
    (0x25c70d9c, &[(0x0204, 0x00000000)]),
    // PA_CL_OBJPRIM_ID_CNTL
    (0x3881201e, &[(0x020d, 0x00000000)]),
    // PA_CL_VTE_CNTL
    (0x09afddaf, &[(0x0206, 0x0000043f)]),
    // PA_SC_AA_CONFIG
    (0x367d63cf, &[(0x02f8, 0x00000000)]),
    // PA_SC_CLIPRECT_RULE
    (0x43707db8, &[(0x0083, 0x0000ffff)]),
    // PA_SC_CONSERVATIVE_RASTERIZATION_CNTL
    (0xf6ae26ba, &[(0x0313, 0x00000000)]),
    // PA_SC_FSR_ENABLE
    (0x1b917652, &[(0x800003fe, 0x00000000)]),
    // PA_SC_HORIZ_GRID
    (0x94b1e4f7, &[(0x00ea, 0x00000000)]),
    // PA_SC_LEFT_VERT_GRID
    (0xe3661b6c, &[(0x00e9, 0x00000000)]),
    // PA_SC_MODE_CNTL_0
    (0x1eb8d73a, &[(0x0292, 0x00000002)]),
    // PA_SC_MODE_CNTL_1
    (0x15051fa3, &[(0x0293, 0x00000000)]),
    // PA_SC_RIGHT_VERT_GRID
    (0x9c51a7f1, &[(0x00e8, 0x00000000)]),
    // PA_SC_WINDOW_OFFSET
    (0xa20efc70, &[(0x0080, 0x00000000)]),
    // PA_STATE_STEREO_X
    (0x0ec09f6e, &[(0x0211, 0x00000000)]),
    // PA_STEREO_CNTL
    (0x34a7d6d3, &[(0x0210, 0x00000000)]),
    // PA_SU_HARDWARE_SCREEN_OFFSET
    (0xce831b94, &[(0x008d, 0x00000000)]),
    // PA_SU_LINE_CNTL
    (0x5cc72a74, &[(0x0282, 0x00000008)]),
    // PA_SU_POINT_MINMAX
    (0x3b77713c, &[(0x0281, 0xffff0000)]),
    // PA_SU_POINT_SIZE
    (0x40f64410, &[(0x0280, 0x00080008)]),
    // PA_SU_POLY_OFFSET_CLAMP
    (0x69441268, &[(0x02df, 0x00000000)]),
    // PA_SU_POLY_OFFSET_DB_FMT_CNTL
    (0x2e418b83, &[(0x02de, 0x000001e9)]),
    // PA_SU_SC_MODE_CNTL
    (0xa00d0c8d, &[(0x0205, 0x00000240)]),
    // PA_SU_SMALL_PRIM_FILTER_CNTL
    (0xb1289fb3, &[(0x020c, 0x00000001)]),
    // PA_SU_VTX_CNTL
    (0x144832fb, &[(0x02f9, 0x0000002d)]),
    // SPI_TMPRING_SIZE
    (0x9890d9fa, &[(0x01ba, 0x00000000)]),
    // VGT_DRAW_PAYLOAD_CNTL
    (0x9016faf1, &[(0x02a6, 0x00000000)]),
    // VGT_GS_MAX_VERT_OUT
    (0x4b73ce27, &[(0x02ce, 0x00000400)]),
    // VGT_GS_OUT_PRIM_TYPE
    (0x5f5a3e7b, &[(0x029b, 0x00000002)]),
    // VGT_LS_HS_CONFIG
    (0xd4af3a51, &[(0x02d6, 0x00000000)]),
    // VGT_PRIMITIVEID_RESET
    (0x6cf4f543, &[(0x02a3, 0xffffffff)]),
    // VGT_PRIMITIVEID_EN
    (0x5fb86ccb, &[(0x02a1, 0x00000000)]),
    // VGT_REUSE_OFF
    (0xedefa188, &[(0x02ad, 0x00000000)]),
    // VGT_SHADER_STAGES_EN
    (0xd0de9ee6, &[(0x02d5, 0x00000000)]),
    // VGT_TESS_DISTRIBUTION
    (0xc5831803, &[(0x02d4, 0x88101000)]),
    // VGT_TF_PARAM
    (0x8e6de84b, &[(0x02db, 0x00000000)]),
    // PA_SC_CENTROID_PRIORITY_0, PA_SC_CENTROID_PRIORITY_1
    (0xd0771662, &[(0x02f5, 0x00000000), (0x02f6, 0x00000000)]),
    // PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0
    (0x569f7444, &[(0x02fe, 0x00000000)]),
    // PA_SC_AA_MASK_X0Y0_X1Y0, PA_SC_AA_MASK_X0Y1_X1Y1
    (0x5c6637cd, &[(0x030e, 0xffffffff), (0x030f, 0xffffffff)]),
    // PA_SC_BINNER_CNTL_0, PA_SC_BINNER_CNTL_1
    (0xcae3e690, &[(0x0311, 0x00000002), (0x0312, 0x03ff0080)]),
    // CB_BLEND_RED, CB_BLEND_BLUE, CB_BLEND_GREEN, CB_BLEND_ALPHA
    (0x43fbd769, &[(0x0105, 0x00000000), (0x0107, 0x00000000), (0x0106, 0x00000000), (0x0108, 0x00000000)]),
    // CB_BLEND0_CONTROL
    (0xef550356, &[(0x01e0, 0x20010001)]),
    // TA_BC_BASE_ADDR, TA_BC_BASE_ADDR_HI
    (0x8f52e279, &[(0x0020, 0x00000000), (0x0021, 0x00000000)]),
    // PA_SC_CLIPRECT_0_TL, PA_SC_CLIPRECT_0_BR
    (0x1f2d8149, &[(0x0084, 0x00000000), (0x0085, 0x20002000)]),
    // CX_NOP
    (0x853d0614, &[(0x800003ff, 0x00000000)]),
    // DB_DEPTH_BOUNDS_MIN, DB_DEPTH_BOUNDS_MAX
    (0x4413c6f9, &[(0x0008, 0x00000000), (0x0009, 0x00000000)]),
    // DB_Z_INFO, DB_STENCIL_INFO, DB_Z_READ_BASE, DB_STENCIL_READ_BASE, DB_Z_WRITE_BASE, DB_STENCIL_WRITE_BASE, DB_Z_READ_BASE_HI, DB_STENCIL_READ_BASE_HI, DB_Z_WRITE_BASE_HI, DB_STENCIL_WRITE_BASE_HI, DB_HTILE_DATA_BASE_HI, DB_DEPTH_VIEW, DB_HTILE_DATA_BASE, DB_DEPTH_SIZE_XY, DB_DEPTH_CLEAR, DB_STENCIL_CLEAR
    (0x67096014, &[(0x0010, 0x80000000), (0x0011, 0x20000000), (0x0012, 0x00000000), (0x0013, 0x00000000), (0x0014, 0x00000000), (0x0015, 0x00000000), (0x001a, 0x00000000), (0x001b, 0x00000000), (0x001c, 0x00000000), (0x001d, 0x00000000), (0x001e, 0x00000000), (0x0002, 0x00000000), (0x0005, 0x00000000), (0x0007, 0x00000000), (0x000b, 0x00000000), (0x000a, 0x00000000)]),
    // PA_SC_FOV_WINDOW_LR, PA_SC_FOV_WINDOW_TB
    (0x88f5e915, &[(0x00eb, 0xff00ff00), (0x00ec, 0x00000000)]),
    // FSR_RECURSIONS0, FSR_RECURSIONS1
    (0x033f1eff, &[(0x800003fc, 0x00000000), (0x800003fd, 0x00000000)]),
    // PA_SC_GENERIC_SCISSOR_TL, PA_SC_GENERIC_SCISSOR_BR
    (0x918106bb, &[(0x0090, 0x80000000), (0x0091, 0x40004000)]),
    // PA_CL_GB_VERT_CLIP_ADJ, PA_CL_GB_VERT_DISC_ADJ, PA_CL_GB_HORZ_CLIP_ADJ, PA_CL_GB_HORZ_DISC_ADJ
    (0x95f0e7ac, &[(0x02fa, 0x4e7e0000), (0x02fb, 0x4e7e0000), (0x02fc, 0x4e7e0000), (0x02fd, 0x4e7e0000)]),
    // PA_SU_POLY_OFFSET_BACK_SCALE, PA_SU_POLY_OFFSET_BACK_OFFSET
    (0xb48cbab2, &[(0x02e2, 0x00000000), (0x02e3, 0x00000000)]),
    // PA_SU_POLY_OFFSET_FRONT_SCALE, PA_SU_POLY_OFFSET_FRONT_OFFSET
    (0x05bb3bc6, &[(0x02e0, 0x00000000), (0x02e1, 0x00000000)]),
    // DB_RENDER_OVERRIDE, DB_RENDER_OVERRIDE2
    (0x94faba07, &[(0x0003, 0x00000000), (0x0004, 0x00000000)]),
    // CB_COLOR0_BASE, CB_COLOR0_VIEW, CB_COLOR0_INFO, CB_COLOR0_ATTRIB, CB_COLOR0_DCC_CONTROL, CB_COLOR0_CMASK, CB_COLOR0_FMASK, CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1, CB_COLOR0_DCC_BASE, CB_COLOR0_BASE_EXT, CB_COLOR0_CMASK_BASE_EXT, CB_COLOR0_FMASK_BASE_EXT, CB_COLOR0_DCC_BASE_EXT, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3
    (0x38e92c91, &[(0x0318, 0x00000000), (0x031b, 0x00000000), (0x031c, 0x00000000), (0x031d, 0x00000000), (0x031e, 0x00000048), (0x031f, 0x00000000), (0x0321, 0x00000000), (0x0323, 0x00000000), (0x0324, 0x00000000), (0x0325, 0x00000000), (0x0390, 0x00000000), (0x0398, 0x00000000), (0x03a0, 0x00000000), (0x03a8, 0x00000000), (0x03b0, 0x00000000), (0x03b8, 0x0006c000)]),
    // PA_SC_SCREEN_SCISSOR_TL, PA_SC_SCREEN_SCISSOR_BR
    (0x0b177b43, &[(0x000c, 0x00000000), (0x000d, 0x40004000)]),
    // SPI_PS_INPUT_CNTL_0
    (0x48531062, &[(0x0191, 0x00000000)]),
    // PA_CL_UCP_0_X, PA_CL_UCP_0_Y, PA_CL_UCP_0_Z, PA_CL_UCP_0_W
    (0xaaa964b9, &[(0x016f, 0x00000000), (0x0170, 0x00000000), (0x0171, 0x00000000), (0x0172, 0x00000000)]),
    // PA_CL_VPORT_XSCALE, PA_CL_VPORT_YSCALE, PA_CL_VPORT_ZSCALE, PA_CL_VPORT_XOFFSET, PA_CL_VPORT_YOFFSET, PA_CL_VPORT_ZOFFSET, PA_SC_VPORT_SCISSOR_0_TL, PA_SC_VPORT_SCISSOR_0_BR, PA_SC_VPORT_ZMIN_0, PA_SC_VPORT_ZMAX_0
    (0x7690af6f, &[(0x010f, 0x4e7e0000), (0x0111, 0x4e7e0000), (0x0113, 0x4e7e0000), (0x0110, 0x00000000), (0x0112, 0x00000000), (0x0114, 0x00000000), (0x0094, 0x80000000), (0x0095, 0x40004000), (0x00b4, 0x00000000), (0x00b5, 0x00000000)]),
    // PA_SC_WINDOW_SCISSOR_TL, PA_SC_WINDOW_SCISSOR_BR
    (0x078d7060, &[(0x0081, 0x80000000), (0x0082, 0x40004000)]),
];

#[rustfmt::skip]
pub(crate) static SH_REG_INFO1: &[RegisterDefaultInfo] = &[
    // COMPUTE_PGM_RSRC1
    (0x5d6e3ec7, &[(0x0212, 0x00000000)]),
    // COMPUTE_PGM_RSRC2
    (0x57e7079a, &[(0x0213, 0x00000000)]),
    // COMPUTE_PGM_RSRC3
    (0x7467fafd, &[(0x0228, 0x00000000)]),
    // COMPUTE_RESOURCE_LIMITS
    (0x9e826b50, &[(0x0215, 0x00000000)]),
    // COMPUTE_TMPRING_SIZE
    (0xdc484f18, &[(0x0218, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC1_GS
    (0x5da8bca3, &[(0x008a, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC1_HS
    (0x5ca726d8, &[(0x010a, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC1_PS
    (0x5dd28360, &[(0x000a, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC2_GS
    (0x57efa0be, &[(0x008b, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC2_HS
    (0x502363d5, &[(0x010b, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC2_PS
    (0x506d14bd, &[(0x000b, 0x00000000)]),
    // COMPUTE_USER_ACCUM_0
    (0xb2609506, &[(0x0224, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC3_HS, SPI_SHADER_PGM_RSRC3_GS, SPI_SHADER_PGM_RSRC3_PS
    (0x9e5cfb8a, &[(0x0107, 0x00000000), (0x0087, 0x00000000), (0x0007, 0x00000000)]),
    // COMPUTE_PGM_LO, COMPUTE_PGM_HI
    (0xc918df3e, &[(0x020c, 0x00000000), (0x020d, 0x00000000)]),
    // SPI_SHADER_PGM_LO_ES, SPI_SHADER_PGM_HI_ES
    (0xc9751c9c, &[(0x00c8, 0x00000000), (0x00c9, 0x00000000)]),
    // SPI_SHADER_PGM_LO_GS, SPI_SHADER_PGM_HI_GS
    (0xc97ef77a, &[(0x0088, 0x00000000), (0x0089, 0x00000000)]),
    // SPI_SHADER_PGM_LO_HS, SPI_SHADER_PGM_HI_HS
    (0xc927c6b9, &[(0x0108, 0x00000000), (0x0109, 0x00000000)]),
    // SPI_SHADER_PGM_LO_LS, SPI_SHADER_PGM_HI_LS
    (0xc92a1ec5, &[(0x0148, 0x00000000), (0x0149, 0x00000000)]),
    // SPI_SHADER_PGM_LO_PS, SPI_SHADER_PGM_HI_PS
    (0xc9e01b31, &[(0x0008, 0x00000000), (0x0009, 0x00000000)]),
    // SH_NOP
    (0x50685f29, &[(0x800002ff, 0x00000000)]),
    // SPI_SHADER_USER_ACCUM_ESGS_0
    (0xb26219ca, &[(0x00b2, 0x00000000)]),
    // SPI_SHADER_USER_ACCUM_LSHS_0
    (0xb25b6cf9, &[(0x0132, 0x00000000)]),
    // SPI_SHADER_USER_ACCUM_PS_0
    (0xb2f86101, &[(0x0032, 0x00000000)]),
    // SPI_SHADER_USER_DATA_ADDR_LO_GS, SPI_SHADER_USER_DATA_ADDR_HI_GS
    (0x07e3b155, &[(0x0082, 0x00000000), (0x0083, 0x00000000)]),
    // SPI_SHADER_USER_DATA_ADDR_LO_HS, SPI_SHADER_USER_DATA_ADDR_HI_HS
    (0x07e383c6, &[(0x0102, 0x00000000), (0x0103, 0x00000000)]),
    // COMPUTE_USER_DATA_0
    (0xbda98653, &[(0x0240, 0x00000000)]),
    // SPI_SHADER_USER_DATA_GS_0
    (0xbdbd1d0f, &[(0x008c, 0x00000000)]),
    // SPI_SHADER_USER_DATA_HS_0
    (0xbd946fd4, &[(0x010c, 0x00000000)]),
    // SPI_SHADER_USER_DATA_PS_0
    (0xbdf02a4c, &[(0x000c, 0x00000000)]),
];

#[rustfmt::skip]
pub(crate) static UC_REG_INFO1: &[RegisterDefaultInfo] = &[
    // GDS_OA_ADDRESS
    (0x19e93e85, &[(0x041f, 0x00000000)]),
    // GDS_OA_CNTL
    (0x3b5c2af3, &[(0x041d, 0x00000000)]),
    // GDS_OA_COUNTER
    (0x47974a35, &[(0x041e, 0x00000000)]),
    // GE_CNTL
    (0x105971c2, &[(0x025b, 0x00000000)]),
    // GE_INDX_OFFSET
    (0x7d137765, &[(0x024a, 0x00000000)]),
    // GE_MULTI_PRIM_IB_RESET_EN
    (0xd187febc, &[(0x024b, 0x00000000)]),
    // GE_STEREO_CNTL
    (0x12f854ac, &[(0x025f, 0x00000000)]),
    // GE_USER_VGPR_EN
    (0x40d49ad1, &[(0x0262, 0x00000000)]),
    // FSR_EXTEND_SUBPIXEL_ROUNDING
    (0x8c0923da, &[(0x80003ff4, 0x00000000)]),
    // TEXTURE_GRADIENT_CONTROL
    (0xbb8df494, &[(0x80003ffd, 0x00000000)]),
    // TEXTURE_GRADIENT_FACTORS
    (0xf6d8a76e, &[(0x0382, 0x40000040)]),
    // VGT_OBJECT_ID
    (0x7620f1e9, &[(0x0248, 0x00000000)]),
    // VGT_PRIMITIVE_TYPE
    (0x9ebfab10, &[(0x0242, 0x00000000)]),
    // TA_CS_BC_BASE_ADDR, TA_CS_BC_BASE_ADDR_HI
    (0x98a09d0e, &[(0x0380, 0x00000000), (0x0381, 0x00000000)]),
    // FSR_ALPHA_VALUE0, FSR_ALPHA_VALUE1
    (0x195d37d2, &[(0x80003ff5, 0x00000000), (0x80003ff6, 0x00000000)]),
    // FSR_CONTROL_POINT0, FSR_CONTROL_POINT1, FSR_CONTROL_POINT2, FSR_CONTROL_POINT3
    (0xf9ec4f85, &[(0x80003ff7, 0x00000000), (0x80003ff8, 0x00000000), (0x80003ff9, 0x00000000), (0x80003ffa, 0x00000000)]),
    // FSR_WINDOW0, FSR_WINDOW1
    (0x4626b750, &[(0x80003ffb, 0x00000000), (0x80003ffc, 0x00000000)]),
    // MEMORY_MAPPING_MASK
    (0x4cc673a0, &[(0x80003ffe, 0x00000000)]),
    // UC_NOP
    (0xde5b3431, &[(0x80003fff, 0x00000000)]),
    // GE_USER_VGPR1
    (0x036ac8a6, &[(0x025c, 0x00000000)]),
];

#[rustfmt::skip]
pub(crate) static CX_REG_INFO2: &[RegisterDefaultInfo] = &[
    // DB_DFSM_CONTROL
    (0x8fb4edb5, &[(0x000e, 0x00000000)]),
    // DB_HTILE_SURFACE
    (0xb994ad29, &[(0x02af, 0x00000000)]),
    // PA_SC_NGG_MODE_CNTL
    (0xd427322f, &[(0x0314, 0x00000000)]),
    // SPI_INTERP_CONTROL_0
    (0xf58fea31, &[(0x01b5, 0x00000000)]),
];

#[rustfmt::skip]
pub(crate) static SH_REG_INFO2: &[RegisterDefaultInfo] = &[
    // COMPUTE_DESTINATION_EN_SE0
    (0x6ac156ef, &[(0x0216, 0x00000000)]),
    // COMPUTE_DESTINATION_EN_SE1
    (0x6ac15610, &[(0x0217, 0x00000000)]),
    // COMPUTE_DESTINATION_EN_SE2
    (0x6ac15009, &[(0x0219, 0x00000000)]),
    // COMPUTE_DESTINATION_EN_SE3
    (0x6ac153ba, &[(0x021a, 0x00000000)]),
    // COMPUTE_DISPATCH_TUNNEL
    (0xbe7dcd73, &[(0x027d, 0x00000000)]),
    // COMPUTE_SHADER_CHKSUM
    (0x0c4b1438, &[(0x022a, 0x00000000)]),
    // COMPUTE_START_X
    (0xdb00d71a, &[(0x0204, 0x00000000)]),
    // COMPUTE_START_Y
    (0xdb00d249, &[(0x0205, 0x00000000)]),
    // COMPUTE_START_Z
    (0xdb00ec60, &[(0x0206, 0x00000000)]),
    // SPI_SHADER_PGM_CHKSUM_GS
    (0x0c4d6fe4, &[(0x0080, 0x00000000)]),
    // SPI_SHADER_PGM_CHKSUM_HS
    (0x0c4a80ef, &[(0x0100, 0x00000000)]),
    // SPI_SHADER_PGM_CHKSUM_PS
    (0x0dd283e7, &[(0x0006, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC4_GS
    (0xc620e68c, &[(0x0081, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC4_HS
    (0xc67efacf, &[(0x0101, 0x00000000)]),
    // SPI_SHADER_PGM_RSRC4_PS
    (0xd9e6d9f7, &[(0x0001, 0x00000000)]),
];

#[rustfmt::skip]
pub(crate) static UC_REG_INFO2: &[RegisterDefaultInfo] = &[
    // VGT_HS_OFFCHIP_PARAM
    (0x31f34b9f, &[(0x024f, 0x00000000)]),
    // UC_NOP
    (0xac0f9e76, &[(0x80003fff, 0x00000000)]),
    // VGT_TF_MEMORY_BASE
    (0x929fd95d, &[(0x0250, 0x00000000)]),
];
