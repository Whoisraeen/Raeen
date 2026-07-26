//! Shader resource data model ("sharps", bind resources, input infos, PS5
//! shader-file header), ported from Kyty (MIT (c) InoriRus).
//!
//! Kyty source: `emulator/include/Emulator/Graphics/Shader.h` L532-1028.
//!
//! C++ bitfield structs are stored as raw dwords with accessor methods (same
//! bit layout); C++ raw pointers inside file-format structs become u64 guest
//! addresses or owned `Vec`s (see each item's doc note).

/// Kyty: Shader.h `ShaderId` (L532). Cache key: equality is on all three
/// fields (`hash0`, `crc32`, `ids`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderId {
    pub hash0: u32,
    pub crc32: u32,
    pub ids: Vec<u32>,
}

/// Kyty: Shader.h `DstSel(x, y, z, w)` (L542).
#[must_use]
pub const fn dst_sel(x: u32, y: u32, z: u32, w: u32) -> u32 {
    x | (y << 3) | (z << 6) | (w << 9)
}

/// Kyty: Shader.h `GetDstSel(swizzle, channel)` (L547).
#[must_use]
pub const fn get_dst_sel(swizzle: u32, channel: u32) -> u8 {
    ((swizzle >> (channel * 3)) & 0x7) as u8
}

/// Kyty: Shader.h `ShaderBufferResource` (L552) — a V# (vertex/buffer
/// descriptor), 4 dwords. Legacy (PS4) accessors: `base44`/`nfmt`/`dfmt`;
/// next-gen (PS5): `base48`/`format`/`out_of_bounds`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderBufferResource {
    pub fields: [u32; 4],
}

impl ShaderBufferResource {
    /// Kyty: `UpdateAddress44` (L556).
    pub fn update_address44(&mut self, gpu_addr: u64) {
        let lo = (gpu_addr & 0xffff_ffff) as u32;
        let hi = (gpu_addr >> 32) as u32;
        self.fields[0] = lo;
        self.fields[1] = (self.fields[1] & 0xffff_f000) | (hi & 0x0000_0fff);
    }

    /// Kyty: `UpdateAddress48` (L564).
    pub fn update_address48(&mut self, gpu_addr: u64) {
        let lo = (gpu_addr & 0xffff_ffff) as u32;
        let hi = (gpu_addr >> 32) as u32;
        self.fields[0] = lo;
        self.fields[1] = (self.fields[1] & 0xffff_0000) | (hi & 0x0000_ffff);
    }

    #[must_use]
    pub const fn stride(&self) -> u16 {
        ((self.fields[1] >> 16) & 0x3FFF) as u16
    }

    #[must_use]
    pub const fn swizzle_enabled(&self) -> bool {
        (self.fields[1] >> 31) & 0x1 == 1
    }

    #[must_use]
    pub const fn num_records(&self) -> u32 {
        self.fields[2]
    }

    #[must_use]
    pub const fn dst_sel_x(&self) -> u8 {
        (self.fields[3] & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_y(&self) -> u8 {
        ((self.fields[3] >> 3) & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_z(&self) -> u8 {
        ((self.fields[3] >> 6) & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_w(&self) -> u8 {
        ((self.fields[3] >> 9) & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_xy(&self) -> u32 {
        self.fields[3] & 0x3F
    }

    #[must_use]
    pub const fn dst_sel_xyz(&self) -> u32 {
        self.fields[3] & 0x1FF
    }

    #[must_use]
    pub const fn dst_sel_xyzw(&self) -> u32 {
        self.fields[3] & 0xFFF
    }

    #[must_use]
    pub const fn add_tid(&self) -> bool {
        (self.fields[3] >> 23) & 0x1 == 1
    }

    /// Next-gen 48-bit base address.
    #[must_use]
    pub const fn base48(&self) -> u64 {
        (self.fields[0] as u64 | ((self.fields[1] as u64) << 32)) & 0xFFFF_FFFF_FFFF
    }

    /// Next-gen format field (7 bits; overlays legacy nfmt+dfmt).
    #[must_use]
    pub const fn format(&self) -> u8 {
        ((self.fields[3] >> 12) & 0x7F) as u8
    }

    #[must_use]
    pub const fn out_of_bounds(&self) -> u8 {
        ((self.fields[3] >> 28) & 0x3) as u8
    }

    /// Legacy 44-bit base address.
    #[must_use]
    pub const fn base44(&self) -> u64 {
        (self.fields[0] as u64 | ((self.fields[1] as u64) << 32)) & 0x0FFF_FFFF_FFFF
    }

    #[must_use]
    pub const fn nfmt(&self) -> u8 {
        ((self.fields[3] >> 12) & 0x7) as u8
    }

    #[must_use]
    pub const fn dfmt(&self) -> u8 {
        ((self.fields[3] >> 15) & 0xF) as u8
    }

    #[must_use]
    pub const fn memory_type(&self) -> u8 {
        (((self.fields[1] >> 7) & 0x60)
            | ((self.fields[3] >> 25) & 0x1c)
            | ((self.fields[1] >> 14) & 0x3)) as u8
    }
}

/// Kyty: Shader.h `ShaderTextureResource` (L597) — a T# (texture descriptor),
/// 8 dwords. Legacy accessors use a 38-bit base; next-gen a 40-bit base.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderTextureResource {
    pub fields: [u32; 8],
}

impl ShaderTextureResource {
    /// Kyty: `UpdateAddress38` (L601).
    pub fn update_address38(&mut self, gpu_addr: u64) {
        let lo = (gpu_addr & 0xffff_ffff) as u32;
        let hi = (gpu_addr >> 32) as u32;
        self.fields[0] = lo;
        self.fields[1] = (self.fields[1] & 0xffff_ffc0) | (hi & 0x0000_003f);
    }

    /// Kyty: `UpdateAddress40` (L609).
    pub fn update_address40(&mut self, gpu_addr: u64) {
        let lo = (gpu_addr & 0xffff_ffff) as u32;
        let hi = (gpu_addr >> 32) as u32;
        self.fields[0] = lo;
        self.fields[1] = (self.fields[1] & 0xffff_ff00) | (hi & 0x0000_00ff);
    }

    #[must_use]
    pub const fn min_lod(&self) -> u16 {
        ((self.fields[1] >> 8) & 0xFFF) as u16
    }

    #[must_use]
    pub const fn dst_sel_x(&self) -> u8 {
        (self.fields[3] & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_y(&self) -> u8 {
        ((self.fields[3] >> 3) & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_z(&self) -> u8 {
        ((self.fields[3] >> 6) & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_w(&self) -> u8 {
        ((self.fields[3] >> 9) & 0x7) as u8
    }

    #[must_use]
    pub const fn dst_sel_xy(&self) -> u32 {
        self.fields[3] & 0x3F
    }

    #[must_use]
    pub const fn dst_sel_xyz(&self) -> u32 {
        self.fields[3] & 0x1FF
    }

    #[must_use]
    pub const fn dst_sel_xyzw(&self) -> u32 {
        self.fields[3] & 0xFFF
    }

    #[must_use]
    pub const fn base_level(&self) -> u8 {
        ((self.fields[3] >> 12) & 0xF) as u8
    }

    #[must_use]
    pub const fn last_level(&self) -> u8 {
        ((self.fields[3] >> 16) & 0xF) as u8
    }

    #[must_use]
    pub const fn tile_mode(&self) -> u8 {
        ((self.fields[3] >> 20) & 0x1F) as u8
    }

    #[must_use]
    pub const fn type_(&self) -> u8 {
        ((self.fields[3] >> 28) & 0xF) as u8
    }

    #[must_use]
    pub const fn depth(&self) -> u16 {
        (self.fields[4] & 0x1FFF) as u16
    }

    // --- next-gen (PS5) accessors ---

    #[must_use]
    pub const fn base40(&self) -> u64 {
        ((self.fields[0] as u64 | ((self.fields[1] as u64) << 32)) & 0xFF_FFFF_FFFF) << 8
    }

    #[must_use]
    pub const fn format(&self) -> u16 {
        ((self.fields[1] >> 20) & 0x1FF) as u16
    }

    #[must_use]
    pub const fn width5(&self) -> u16 {
        (((self.fields[1] >> 30) & 0x3) | ((self.fields[2] & 0xFFF) << 2)) as u16
    }

    #[must_use]
    pub const fn height5(&self) -> u16 {
        ((self.fields[2] >> 14) & 0x3FFF) as u16
    }

    #[must_use]
    pub const fn bc_swizzle(&self) -> u8 {
        ((self.fields[3] >> 25) & 0x7) as u8
    }

    #[must_use]
    pub const fn base_array5(&self) -> u16 {
        ((self.fields[4] >> 16) & 0x1FFF) as u16
    }

    #[must_use]
    pub const fn array_pitch(&self) -> u8 {
        (self.fields[5] & 0xF) as u8
    }

    #[must_use]
    pub const fn max_mip(&self) -> u8 {
        ((self.fields[5] >> 4) & 0xF) as u8
    }

    #[must_use]
    pub const fn min_lod_warn5(&self) -> u16 {
        ((self.fields[5] >> 8) & 0xFFF) as u16
    }

    #[must_use]
    pub const fn perf_mod5(&self) -> u8 {
        ((self.fields[5] >> 20) & 0x7) as u8
    }

    #[must_use]
    pub const fn corner_sample(&self) -> bool {
        (self.fields[5] >> 23) & 0x1 == 1
    }

    #[must_use]
    pub const fn mip_stats_cnt_en(&self) -> bool {
        (self.fields[5] >> 25) & 0x1 == 1
    }

    #[must_use]
    pub const fn prt_def_color(&self) -> bool {
        (self.fields[5] >> 26) & 0x1 == 1
    }

    #[must_use]
    pub const fn mip_stats_cnt_id(&self) -> u8 {
        (self.fields[6] & 0xFF) as u8
    }

    #[must_use]
    pub const fn msaa_depth(&self) -> bool {
        (self.fields[6] >> 10) & 0x1 == 1
    }

    #[must_use]
    pub const fn max_uncomp_blk_size(&self) -> u8 {
        ((self.fields[6] >> 15) & 0x3) as u8
    }

    #[must_use]
    pub const fn max_comp_blk_size(&self) -> u8 {
        ((self.fields[6] >> 17) & 0x3) as u8
    }

    #[must_use]
    pub const fn meta_pipe_aligned(&self) -> bool {
        (self.fields[6] >> 19) & 0x1 == 1
    }

    #[must_use]
    pub const fn write_compress(&self) -> bool {
        (self.fields[6] >> 20) & 0x1 == 1
    }

    #[must_use]
    pub const fn meta_compress(&self) -> bool {
        (self.fields[6] >> 21) & 0x1 == 1
    }

    #[must_use]
    pub const fn dcc_alpha_pos(&self) -> bool {
        (self.fields[6] >> 22) & 0x1 == 1
    }

    #[must_use]
    pub const fn dcc_color_transf(&self) -> bool {
        (self.fields[6] >> 23) & 0x1 == 1
    }

    #[must_use]
    pub const fn meta_addr(&self) -> u64 {
        ((self.fields[6] >> 24) & 0xFF) as u64 | ((self.fields[7] as u64) << 8)
    }

    // --- legacy (PS4) accessors ---

    #[must_use]
    pub const fn base38(&self) -> u64 {
        ((self.fields[0] as u64 | ((self.fields[1] as u64) << 32)) & 0x3F_FFFF_FFFF) << 8
    }

    #[must_use]
    pub const fn dfmt(&self) -> u8 {
        ((self.fields[1] >> 20) & 0x3F) as u8
    }

    #[must_use]
    pub const fn nfmt(&self) -> u8 {
        ((self.fields[1] >> 26) & 0xF) as u8
    }

    #[must_use]
    pub const fn width4(&self) -> u16 {
        (self.fields[2] & 0x3FFF) as u16
    }

    #[must_use]
    pub const fn height4(&self) -> u16 {
        ((self.fields[2] >> 14) & 0x3FFF) as u16
    }

    #[must_use]
    pub const fn perf_mod(&self) -> u8 {
        ((self.fields[2] >> 28) & 0x7) as u8
    }

    #[must_use]
    pub const fn interlaced(&self) -> bool {
        (self.fields[2] >> 31) & 0x1 == 1
    }

    #[must_use]
    pub const fn pow2_pad(&self) -> bool {
        (self.fields[3] >> 25) & 0x1 == 1
    }

    #[must_use]
    pub const fn pitch(&self) -> u16 {
        ((self.fields[4] >> 13) & 0x3FFF) as u16
    }

    #[must_use]
    pub const fn base_array(&self) -> u16 {
        (self.fields[5] & 0x1FFF) as u16
    }

    #[must_use]
    pub const fn last_array(&self) -> u16 {
        ((self.fields[5] >> 13) & 0x1FFF) as u16
    }

    #[must_use]
    pub const fn min_lod_warn(&self) -> u16 {
        (self.fields[6] & 0xFFF) as u16
    }

    #[must_use]
    pub const fn lod_hdw_cnt_en(&self) -> bool {
        (self.fields[6] >> 20) & 0x1 == 1
    }

    #[must_use]
    pub const fn counter_bank_id(&self) -> u8 {
        ((self.fields[6] >> 12) & 0xFF) as u8
    }

    #[must_use]
    pub const fn memory_type(&self) -> u8 {
        (((self.fields[1] >> 6) & 0x3)
            | ((self.fields[1] >> 30) << 2)
            | (if self.fields[3] & 0x0400_0000 == 0 {
                0x60
            } else {
                0x10
            })) as u8
    }
}

/// Kyty: Shader.h `ShaderSamplerResource` (L675) — an S# (sampler
/// descriptor), 4 dwords.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSamplerResource {
    pub fields: [u32; 4],
}

impl ShaderSamplerResource {
    /// Kyty: `UpdateIndex` (L679).
    pub fn update_index(&mut self, index: u32) {
        self.fields[0] = index;
    }

    #[must_use]
    pub const fn clamp_x(&self) -> u8 {
        (self.fields[0] & 0x7) as u8
    }

    #[must_use]
    pub const fn clamp_y(&self) -> u8 {
        ((self.fields[0] >> 3) & 0x7) as u8
    }

    #[must_use]
    pub const fn clamp_z(&self) -> u8 {
        ((self.fields[0] >> 6) & 0x7) as u8
    }

    #[must_use]
    pub const fn max_aniso_ratio(&self) -> u8 {
        ((self.fields[0] >> 9) & 0x7) as u8
    }

    #[must_use]
    pub const fn depth_compare_func(&self) -> u8 {
        ((self.fields[0] >> 12) & 0x7) as u8
    }

    #[must_use]
    pub const fn force_unorm_coords(&self) -> bool {
        (self.fields[0] >> 15) & 0x1 == 1
    }

    #[must_use]
    pub const fn aniso_threshold(&self) -> u8 {
        ((self.fields[0] >> 16) & 0x7) as u8
    }

    #[must_use]
    pub const fn force_degamma(&self) -> bool {
        (self.fields[0] >> 20) & 0x1 == 1
    }

    #[must_use]
    pub const fn aniso_bias(&self) -> u8 {
        ((self.fields[0] >> 21) & 0x3F) as u8
    }

    #[must_use]
    pub const fn trunc_coord(&self) -> bool {
        (self.fields[0] >> 27) & 0x1 == 1
    }

    #[must_use]
    pub const fn disable_cube_wrap(&self) -> bool {
        (self.fields[0] >> 28) & 0x1 == 1
    }

    #[must_use]
    pub const fn filter_mode(&self) -> u8 {
        ((self.fields[0] >> 29) & 0x3) as u8
    }

    #[must_use]
    pub const fn min_lod(&self) -> u16 {
        (self.fields[1] & 0xFFF) as u16
    }

    #[must_use]
    pub const fn max_lod(&self) -> u16 {
        ((self.fields[1] >> 12) & 0xFFF) as u16
    }

    #[must_use]
    pub const fn perf_mip(&self) -> u8 {
        ((self.fields[1] >> 24) & 0xF) as u8
    }

    #[must_use]
    pub const fn perf_z(&self) -> u8 {
        ((self.fields[1] >> 28) & 0xF) as u8
    }

    #[must_use]
    pub const fn lod_bias(&self) -> u16 {
        (self.fields[2] & 0x3FFF) as u16
    }

    #[must_use]
    pub const fn lod_bias_sec(&self) -> u8 {
        ((self.fields[2] >> 14) & 0x3F) as u8
    }

    #[must_use]
    pub const fn xy_mag_filter(&self) -> u8 {
        ((self.fields[2] >> 20) & 0x3) as u8
    }

    #[must_use]
    pub const fn xy_min_filter(&self) -> u8 {
        ((self.fields[2] >> 22) & 0x3) as u8
    }

    #[must_use]
    pub const fn z_filter(&self) -> u8 {
        ((self.fields[2] >> 24) & 0x3) as u8
    }

    #[must_use]
    pub const fn mip_filter(&self) -> u8 {
        ((self.fields[2] >> 26) & 0x3) as u8
    }

    #[must_use]
    pub const fn border_color_ptr(&self) -> u16 {
        (self.fields[3] & 0xFFF) as u16
    }

    #[must_use]
    pub const fn border_color_type(&self) -> u8 {
        ((self.fields[3] >> 30) & 0x3) as u8
    }

    /// Next-gen bit 31 of dword 0.
    #[must_use]
    pub const fn skip_degamma(&self) -> bool {
        (self.fields[0] >> 31) & 0x1 == 1
    }

    /// Kyty aliases `PointPreclamp`/`AnisoOverride`/`BlendZeroPrt` all read
    /// bit 28 of dword 3 (Shader.h L707-709 — kept sic).
    #[must_use]
    pub const fn point_preclamp(&self) -> bool {
        (self.fields[3] >> 28) & 0x1 == 1
    }

    #[must_use]
    pub const fn aniso_override(&self) -> bool {
        (self.fields[3] >> 28) & 0x1 == 1
    }

    #[must_use]
    pub const fn blend_zero_prt(&self) -> bool {
        (self.fields[3] >> 28) & 0x1 == 1
    }

    #[must_use]
    pub const fn mc_coord_trunc(&self) -> bool {
        (self.fields[0] >> 19) & 0x1 == 1
    }
}

/// Kyty: Shader.h `ShaderGdsResource` (L714).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderGdsResource {
    pub field: u32,
}

impl ShaderGdsResource {
    #[must_use]
    pub const fn base(&self) -> u16 {
        ((self.field >> 16) & 0xFFFF) as u16
    }

    #[must_use]
    pub const fn size(&self) -> u16 {
        (self.field & 0xFFFF) as u16
    }
}

/// Kyty: Shader.h `ShaderDirectSgprResource` (L722).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderDirectSgprResource {
    pub field: u32,
}

/// Kyty: Shader.h `ShaderExtendedResource` (L727) — EUD base pointer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderExtendedResource {
    pub fields: [u32; 2],
}

impl ShaderExtendedResource {
    /// Kyty: `UpdateAddress` (L731).
    pub fn update_address(&mut self, gpu_addr: u64) {
        self.fields[0] = (gpu_addr & 0xffff_ffff) as u32;
        self.fields[1] = (gpu_addr >> 32) as u32;
    }

    #[must_use]
    pub const fn base(&self) -> u64 {
        self.fields[0] as u64 | ((self.fields[1] as u64) << 32)
    }
}

/// Kyty: Shader.h `ShaderVertexInputBuffer` (L742).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderVertexInputBuffer {
    pub addr: u64,
    pub stride: u32,
    pub num_records: u32,
    /// Gen5 attribute fetch selector: 0 = per-vertex, 1 = per-instance.
    pub fetch_index: u32,
    pub attr_num: i32,
    pub attr_indices: [i32; Self::ATTR_MAX],
    pub attr_offsets: [u32; Self::ATTR_MAX],
}

impl ShaderVertexInputBuffer {
    pub const ATTR_MAX: usize = 16;
}

/// Kyty: Shader.h `ShaderVertexDestination` (L754).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShaderVertexDestination {
    pub register_start: i32,
    pub registers_num: i32,
    /// Gen5 attribute fetch selector: 0 = vertex index, 1 = instance index.
    pub fetch_index: u32,
    /// Attrib-table index (`ShaderSemantic::semantic()`) this resource came
    /// from. Beyond Kyty, which stores resources by array POSITION and then
    /// looks them up by attrib id in `Recompile_Fetch` — the two agree only
    /// while the semantics table is identity-mapped. Minecraft's is not
    /// (measured: positions 0,1,2 carry semantics 0,2,3), so the by-position
    /// read returned another attribute's V# for one id and an unwritten slot
    /// for another. Recording the semantic lets the lookup resolve by id.
    /// `-1` marks a slot that was never populated.
    pub semantic: i32,
}

impl ShaderVertexDestination {
    /// Kyty leaves unpopulated slots zeroed, which is indistinguishable from
    /// "semantic 0". `Default` therefore cannot be derived for `semantic`.
    pub const UNSET_SEMANTIC: i32 = -1;
}

impl Default for ShaderVertexDestination {
    fn default() -> Self {
        Self {
            register_start: 0,
            registers_num: 0,
            fetch_index: 0,
            semantic: Self::UNSET_SEMANTIC,
        }
    }
}

/// Kyty: Shader.h `ShaderStorageUsage` (L760).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ShaderStorageUsage {
    #[default]
    Unknown,
    Constant,
    ReadOnly,
    ReadWrite,
}

/// Kyty: Shader.h `ShaderTextureUsage` (L768).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ShaderTextureUsage {
    #[default]
    Unknown,
    ReadOnly,
    ReadWrite,
}

/// Kyty: Shader.h `ShaderStorageResources` (L775).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderStorageResources {
    pub buffers: [ShaderBufferResource; Self::BUFFERS_MAX],
    pub usages: [ShaderStorageUsage; Self::BUFFERS_MAX],
    /// Smallest byte prefix proven sufficient for this binding's scalar
    /// constant-buffer loads. Zero means the shader uses a dynamic offset (or
    /// no access was decoded), so the full descriptor extent is required.
    ///
    /// This is bind-time upload metadata only; it does not alter the SPIR-V
    /// descriptor ABI or shader-cache identity.
    pub required_bytes: [u32; Self::BUFFERS_MAX],
    pub slots: [i32; Self::BUFFERS_MAX],
    pub start_register: [i32; Self::BUFFERS_MAX],
    pub extended: [bool; Self::BUFFERS_MAX],
    pub buffers_num: i32,
    pub binding_index: i32,
}

impl ShaderStorageResources {
    pub const BUFFERS_MAX: usize = 16;
}

/// Kyty: Shader.h `ShaderTextureDescriptor` (L788).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderTextureDescriptor {
    pub texture: ShaderTextureResource,
    pub usage: ShaderTextureUsage,
    pub slot: i32,
    pub start_register: i32,
    pub extended: bool,
    pub textures2d_without_sampler: bool,
}

/// Kyty: Shader.h `ShaderTextureResources` (L798).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderTextureResources {
    pub desc: [ShaderTextureDescriptor; Self::RES_MAX],
    pub textures_num: i32,
    pub textures2d_sampled_num: i32,
    pub textures2d_storage_num: i32,
    pub binding_sampled_index: i32,
    pub binding_storage_index: i32,
}

impl ShaderTextureResources {
    pub const RES_MAX: usize = 16;
}

/// Kyty: Shader.h `ShaderSamplerResources` (L810).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSamplerResources {
    pub samplers: [ShaderSamplerResource; Self::RES_MAX],
    pub slots: [i32; Self::RES_MAX],
    pub start_register: [i32; Self::RES_MAX],
    pub extended: [bool; Self::RES_MAX],
    pub samplers_num: i32,
    pub binding_index: i32,
}

impl ShaderSamplerResources {
    pub const RES_MAX: usize = 16;
}

/// Kyty: Shader.h `ShaderGdsResources` (L822).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderGdsResources {
    pub pointers: [ShaderGdsResource; Self::POINTERS_MAX],
    pub slots: [i32; Self::POINTERS_MAX],
    pub start_register: [i32; Self::POINTERS_MAX],
    pub extended: [bool; Self::POINTERS_MAX],
    pub pointers_num: i32,
    pub binding_index: i32,
}

impl ShaderGdsResources {
    pub const POINTERS_MAX: usize = 1;
}

/// Kyty: Shader.h `ShaderDirectSgprsResources` (L834).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderDirectSgprsResources {
    pub sgprs: [ShaderDirectSgprResource; Self::SGPRS_MAX],
    pub start_register: [i32; Self::SGPRS_MAX],
    pub sgprs_num: i32,
}

impl ShaderDirectSgprsResources {
    // Kyty caps this at 4 (PS4 stages never left more than a handful of user
    // SGPRs unconsumed). Gen5 CS shaders keep whole SRT blocks as raw user
    // data — up to the full 32-register file — so the cap is the file size.
    pub const SGPRS_MAX: usize = 32;
}

/// Kyty: Shader.h `ShaderExtendedResources` (L843).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderExtendedResources {
    pub used: bool,
    pub slot: i32,
    pub start_register: i32,
    pub data: ShaderExtendedResource,
}

/// Beyond Kyty — SharpEmu port: the raw EUD-window fallback binding.
///
/// SharpEmu has no captured-descriptor-table concept: every scalar load —
/// EUD, SRT, anything — is a dispatch-time guest-memory read, and loads the
/// evaluator cannot resolve statically become a pooled guest-memory window
/// bound as an SSBO (`reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs:1939-1980`, consumed by
/// `Gen5SpirvTranslator.cs:2183-2236`). This is the minimal analogue for the
/// one measured refusal class: an `s_load_dwordx2/x4/x8` off the EUD base
/// pair whose dword is NOT a captured descriptor field (195 refused
/// ASTRO.BOT compute dispatches/run). Captured descriptors keep the
/// push-constant path — this group is an additive fallback, not a rewrite.
///
/// `shader_detect_eud_raw_window` fills it after `shader_get_input_info_*`;
/// the recompiler lowers uncovered dwords to clamped reads of the `%eud_raw`
/// SSBO; the dispatch path snapshots the guest window behind the EUD base
/// pointer into that SSBO (`raeen-gpu` `prepare_stage_binding`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderEudRawResources {
    /// A raw-window read exists: declare + bind the `%eud_raw` SSBO.
    pub used: bool,
    /// Vulkan binding index of `%eud_raw` (next index after every group
    /// `shader_calc_binding_indices` assigned).
    pub binding_index: i32,
    /// Minimum window size in dwords: highest constant dword index any raw
    /// `s_load` addresses, plus one. The host may bind MORE (SharpEmu's
    /// 256 KiB minimum) or LESS (unreadable tail) — the recompiled reads
    /// clamp against the bound size and return 0 beyond it.
    pub required_dwords: u32,
}

/// Beyond Kyty (SharpEmu PR #587): the guest-memory window backing the
/// FLAT-class (FLAT / GLOBAL) direct-address load/stores. Where a V# storage
/// buffer is a bounded descriptor, a FLAT op carries a complete 64-bit guest
/// pointer (or an SGPR base pair plus a per-lane offset) and reads/writes guest
/// memory directly — SharpEmu's "global memory binding"
/// (`Gen5SpirvTranslator.cs`). The recompiler serves it from a single uint
/// runtime-array SSBO (`%global_mem`) whose first two dwords are the window's
/// guest base address (host-filled at bind time) and whose remaining dwords are
/// the window contents; the shader converts a 64-bit address to a dword index
/// by `((addr_lo - base_lo) >> 2)` (32-bit subtraction, exactly as SharpEmu's
/// `ISub` does — the wrap absorbs any carry, and the window is < 4 GiB). Reads
/// past the bound length clamp to 0 and stores past it drop, matching RDNA
/// out-of-bounds behaviour. `used`/`binding_index` are assigned by
/// `shader_detect_flat_global_window` after every other resource group.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderGlobalMemResources {
    /// A FLAT-class op is present: declare + bind the `%global_mem` SSBO.
    pub used: bool,
    /// Vulkan binding index of `%global_mem` (after the raw-EUD fallback).
    pub binding_index: i32,
}

/// Beyond Kyty: one `s_load_dword{,x2,x4,x8,x16}` whose base pointer is built
/// **PC-relative** (`s_getpc_b64` + optional `s_add_u32 <imm>`) rather than
/// from user data or the EUD — the shader loading its own embedded constant
/// table. The absolute address is a compile-time constant (the getpc
/// materializes it; see `parse.rs` S_GETPC_B64), so the loaded dwords are
/// known at recompile time and materialized as SPIR-V constants directly into
/// the destination SGPRs. Measured on ASTRO.BOT vertex shaders.
///
/// `shader_detect_embedded_constant_loads` reads the values out of guest
/// memory during analysis (the recompiler has no raw shader bytes); the
/// recompiler's `sload_dword_extended` matches by `pc` and emits the stores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderEmbeddedConstantLoad {
    /// PC of the `s_load` instruction this capture belongs to (the recompiler
    /// matches on `ShaderInstruction::pc`).
    pub pc: u32,
    /// Dwords fetched: 2, 4, 8, or 16.
    pub dwords_num: u32,
    /// The embedded constant dwords read from the shader binary at the
    /// PC-relative address (only `dwords_num` entries are meaningful).
    pub values: [u32; Self::VALUES_MAX],
}

impl ShaderEmbeddedConstantLoad {
    /// Largest SMRD load width (`s_load_dwordx16`).
    pub const VALUES_MAX: usize = 16;
}

/// Beyond Kyty: the set of PC-relative embedded-constant scalar loads a stage
/// performs (see [`ShaderEmbeddedConstantLoad`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderEmbeddedConstantLoads {
    pub loads: [ShaderEmbeddedConstantLoad; Self::LOADS_MAX],
    pub loads_num: i32,
}

impl ShaderEmbeddedConstantLoads {
    pub const LOADS_MAX: usize = 8;

    /// The captured dwords for the `s_load` at `pc`, if any.
    #[must_use]
    pub fn find(&self, pc: u32) -> Option<&ShaderEmbeddedConstantLoad> {
        self.loads[..self.loads_num.max(0) as usize]
            .iter()
            .find(|l| l.pc == pc)
    }
}

/// Beyond Kyty: one `offen` MUBUF buffer load whose buffer descriptor (V#) is
/// **constructed inside the shader** — its base pointer built PC-relative
/// (`s_getpc_b64` + the 64-bit add idiom), the descriptor words set by
/// immediate moves — pointing at the shader's own embedded vertex data. The
/// descriptor is never a user-data / captured descriptor, so the usual
/// storage-buffer path finds nothing (`buffers_num == 0`) and refuses the load.
///
/// The embedded data is static (baked into the shader binary), so
/// `shader_detect_embedded_buffer_fetch` snapshots a window of it from guest
/// memory at analysis time and the recompiler serves the runtime-indexed
/// (`base + voffset`) read from those constants — a select over the captured
/// window keyed by the per-lane byte offset. Measured on the ASTRO.BOT
/// full-screen-triangle vertex shader (embedded verts `(-1,-1),(3,-1),(-1,3)`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShaderEmbeddedBufferFetch {
    /// PC of the buffer-load instruction this capture belongs to.
    pub pc: u32,
    /// The load's immediate byte offset (MUBUF `offset`, folded soffset).
    pub inst_offset: u32,
    /// Dwords the load fetches (1, 2, 3, or 4).
    pub dwords_num: u32,
    /// Length (in dwords) of the captured window that is meaningful.
    pub window_len: u32,
    /// Embedded data snapshot starting at the in-shader V# base address.
    pub window: [u32; Self::WINDOW_MAX],
}

impl ShaderEmbeddedBufferFetch {
    /// Cap on the captured window. Keeps the select-chain small and every
    /// window index within the seeded `%uint_0..=32` constants; a larger
    /// embedded buffer is left to the recompiler's refusal.
    pub const WINDOW_MAX: usize = 32;
}

impl Default for ShaderEmbeddedBufferFetch {
    fn default() -> Self {
        Self {
            pc: 0,
            inst_offset: 0,
            dwords_num: 0,
            window_len: 0,
            window: [0; Self::WINDOW_MAX],
        }
    }
}

/// Beyond Kyty: the set of in-shader-V# `offen` buffer loads a stage performs
/// (see [`ShaderEmbeddedBufferFetch`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderEmbeddedBufferFetches {
    pub loads: [ShaderEmbeddedBufferFetch; Self::LOADS_MAX],
    pub loads_num: i32,
}

impl ShaderEmbeddedBufferFetches {
    pub const LOADS_MAX: usize = 8;

    /// The captured embedded fetch for the buffer load at `pc`, if any.
    #[must_use]
    pub fn find(&self, pc: u32) -> Option<&ShaderEmbeddedBufferFetch> {
        self.loads[..self.loads_num.max(0) as usize]
            .iter()
            .find(|l| l.pc == pc)
    }
}

/// Kyty: Shader.h `ShaderBindResources` (L851). Aggregated per-stage binding
/// info: push-constant window plus every resource group.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderBindResources {
    pub push_constant_offset: u32,
    pub push_constant_size: u32,
    pub descriptor_set_slot: u32,
    pub storage_buffers: ShaderStorageResources,
    pub textures2d: ShaderTextureResources,
    pub samplers: ShaderSamplerResources,
    pub gds_pointers: ShaderGdsResources,
    pub direct_sgprs: ShaderDirectSgprsResources,
    pub extended: ShaderExtendedResources,
    /// Beyond Kyty (SharpEmu port): raw EUD-window fallback for scalar loads
    /// of EUD dwords no captured descriptor covers.
    pub eud_raw: ShaderEudRawResources,
    /// Beyond Kyty (SharpEmu PR #587): the `%global_mem` window backing
    /// FLAT-class (FLAT / GLOBAL) direct-address memory ops.
    pub global_mem: ShaderGlobalMemResources,
    /// Beyond Kyty: PC-relative embedded-constant scalar loads (the shader
    /// reading its own baked constant table).
    pub embedded_constant_loads: ShaderEmbeddedConstantLoads,
    /// Beyond Kyty: `offen` buffer loads through an in-shader-constructed V#
    /// that points at the shader's own embedded vertex data.
    pub embedded_buffer_fetches: ShaderEmbeddedBufferFetches,
}

/// Kyty: Shader.h `ShaderVertexInputInfo` (L864).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderVertexInputInfo {
    pub resources: [ShaderBufferResource; Self::RES_MAX],
    pub resources_dst: [ShaderVertexDestination; Self::RES_MAX],
    pub buffers: [ShaderVertexInputBuffer; Self::RES_MAX],
    pub bind: ShaderBindResources,
    pub resources_num: i32,
    pub fetch_shader_reg: i32,
    pub fetch_attrib_reg: i32,
    pub fetch_buffer_reg: i32,
    pub buffers_num: i32,
    pub export_count: i32,
    pub fetch_external: bool,
    pub fetch_embedded: bool,
    pub fetch_inline: bool,
    pub gs_prolog: bool,
}

impl ShaderVertexInputInfo {
    pub const RES_MAX: usize = 16;
}

/// Kyty: Shader.h `ShaderComputeInputInfo` (L884).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderComputeInputInfo {
    pub threads_num: [u32; 3],
    pub group_id: [bool; 3],
    pub thread_ids_num: i32,
    pub workgroup_register: i32,
    /// Beyond Kyty: LDS allocation in dwords, decoded from
    /// `COMPUTE_PGM_RSRC2.LDS_SIZE` (128-dword granules). Sizes the
    /// `%lds` Workgroup array backing `ds_write_b32`/`ds_read_b32`.
    pub lds_size_dw: u32,
    pub bind: ShaderBindResources,
}

/// Kyty: Shader.h `ShaderPixelInputInfo` (L893).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderPixelInputInfo {
    pub interpolator_settings: [u32; 32],
    pub input_num: u32,
    pub target_output_mode: [u8; 8],
    pub ps_pos_xy: bool,
    pub ps_pixel_kill_enable: bool,
    pub ps_early_z: bool,
    pub ps_execute_on_noop: bool,
    pub bind: ShaderBindResources,
}

/// Kyty: Shader.h `ShaderSharp` (L905) — `offset_dw:15, size:1` packed in a
/// u16 (bitfields stored raw, accessed by method).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSharp {
    pub raw: u16,
}

impl ShaderSharp {
    #[must_use]
    pub const fn new(offset_dw: u16, size: u16) -> Self {
        Self {
            raw: (offset_dw & 0x7FFF) | ((size & 0x1) << 15),
        }
    }

    #[must_use]
    pub const fn offset_dw(&self) -> u16 {
        self.raw & 0x7FFF
    }

    #[must_use]
    pub const fn size(&self) -> u16 {
        self.raw >> 15
    }
}

/// Kyty: Shader.h `ShaderUserData` (L911) — PS5 user-data mapping tables.
///
/// Deviation: Kyty stores raw pointers plus explicit counts
/// (`direct_resource_count`, `sharp_resource_count[4]`); the port owns the
/// tables as `Vec`s whose lengths are the counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderUserData {
    pub direct_resource_offset: Vec<u16>,
    pub sharp_resource_offset: [Vec<ShaderSharp>; 4],
    pub eud_size_dw: u16,
    pub srt_size_dw: u16,
}

/// Kyty: Shader.h `ShaderRegisterRange` (L921).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderRegisterRange {
    pub start: u16,
    pub end: u16,
}

/// Kyty: Shader.h `ShaderDrawModifier` (L927) — two raw dwords of bitfields.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderDrawModifier {
    pub raw: [u32; 2],
}

impl ShaderDrawModifier {
    #[must_use]
    pub const fn enbl_start_vertex_offset(&self) -> bool {
        self.raw[0] & 0x1 == 1
    }

    #[must_use]
    pub const fn enbl_start_index_offset(&self) -> bool {
        (self.raw[0] >> 1) & 0x1 == 1
    }

    #[must_use]
    pub const fn enbl_start_instance_offset(&self) -> bool {
        (self.raw[0] >> 2) & 0x1 == 1
    }

    #[must_use]
    pub const fn enbl_draw_index(&self) -> bool {
        (self.raw[0] >> 3) & 0x1 == 1
    }

    #[must_use]
    pub const fn enbl_user_vgprs(&self) -> bool {
        (self.raw[0] >> 4) & 0x1 == 1
    }

    #[must_use]
    pub const fn render_target_slice_offset(&self) -> u8 {
        ((self.raw[0] >> 5) & 0x7) as u8
    }

    #[must_use]
    pub const fn fuse_draws(&self) -> bool {
        (self.raw[0] >> 8) & 0x1 == 1
    }

    #[must_use]
    pub const fn compiler_flags(&self) -> u32 {
        self.raw[0] >> 9
    }

    #[must_use]
    pub const fn is_default(&self) -> bool {
        self.raw[1] & 0x1 == 1
    }
}

/// Kyty: Shader.h `ShaderRegister` (L941).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderRegister {
    pub offset: u32,
    pub value: u32,
}

/// Kyty: Shader.h `ShaderSpecialRegs` (L947).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSpecialRegs {
    pub ge_cntl: ShaderRegister,
    pub vgt_shader_stages_en: ShaderRegister,
    pub dispatch_modifier: u32,
    pub user_data_range: ShaderRegisterRange,
    pub draw_modifier: ShaderDrawModifier,
    pub vgt_gs_out_prim_type: ShaderRegister,
    pub ge_user_vgpr_en: ShaderRegister,
}

/// Kyty: Shader.h `ShaderSemantic` (L958) — one raw dword of bitfields
/// describing a PS5 input/output semantic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSemantic {
    pub raw: u32,
}

impl ShaderSemantic {
    #[must_use]
    pub const fn semantic(&self) -> u32 {
        self.raw & 0xFF
    }

    #[must_use]
    pub const fn hardware_mapping(&self) -> u32 {
        (self.raw >> 8) & 0xFF
    }

    #[must_use]
    pub const fn size_in_elements(&self) -> u32 {
        (self.raw >> 16) & 0xF
    }

    #[must_use]
    pub const fn is_f16(&self) -> u32 {
        (self.raw >> 20) & 0x3
    }

    #[must_use]
    pub const fn is_flat_shaded(&self) -> bool {
        (self.raw >> 22) & 0x1 == 1
    }

    #[must_use]
    pub const fn is_linear(&self) -> bool {
        (self.raw >> 23) & 0x1 == 1
    }

    #[must_use]
    pub const fn is_custom(&self) -> bool {
        (self.raw >> 24) & 0x1 == 1
    }

    #[must_use]
    pub const fn static_vb_index(&self) -> bool {
        (self.raw >> 25) & 0x1 == 1
    }

    #[must_use]
    pub const fn static_attribute(&self) -> bool {
        (self.raw >> 26) & 0x1 == 1
    }

    #[must_use]
    pub const fn default_value(&self) -> u32 {
        (self.raw >> 28) & 0x3
    }

    #[must_use]
    pub const fn default_value_hi(&self) -> u32 {
        (self.raw >> 30) & 0x3
    }
}

/// Kyty: Shader.h `Shader` (L974) — the PS5 shader-file header.
///
/// Deviation: renamed to `ShaderFileHeader` (the module already has a
/// `shader` namespace and `ShaderCode`); Kyty's raw pointers
/// (`user_data`, `code`, `cx_registers`, ...) are stored as u64 guest
/// addresses.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderFileHeader {
    pub file_header: u32,
    pub version: u32,
    pub user_data: u64,
    pub code: u64,
    pub cx_registers: u64,
    pub sh_registers: u64,
    pub specials: u64,
    pub input_semantics: u64,
    pub output_semantics: u64,
    pub header_size: u32,
    pub shader_size: u32,
    pub embedded_constant_buffer_size_dqw: u32,
    pub target: u32,
    pub num_input_semantics: u32,
    pub scratch_size_dw_per_thread: u16,
    pub num_output_semantics: u16,
    pub special_sizes_bytes: u16,
    pub type_: u8,
    pub num_cx_registers: u8,
    pub num_sh_registers: u8,
}

/// Kyty: Shader.h `ShaderMappedData` (L998) — what `ShaderMapUserData`
/// registers per shader address on PS5.
///
/// Deviation: owned data instead of raw pointers;
/// `num_input_semantics` is `input_semantics.len()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderMappedData {
    pub user_data: Option<ShaderUserData>,
    pub input_semantics: Vec<ShaderSemantic>,
}
#[cfg(test)]
mod tests {
    use super::*;

    // ---- 2. Sharp accessors (hand-computed from Shader.h bit layouts) ----

    #[test]
    fn buffer_resource_accessors() {
        // Kyty: Shader.h ShaderBufferResource (L552).
        // f1 = base_hi 0x45 | stride 0x123 << 16 | swizzle << 31.
        // f3 = DstSel(4,5,6,7) | nfmt 5 << 12 | dfmt 10 << 15 | add_tid << 23
        //      | out_of_bounds 2 << 28.
        let r = ShaderBufferResource {
            fields: [0x89AB_CDEF, 0x8123_0045, 1000, 0x2085_5FAC],
        };
        assert_eq!(r.stride(), 0x123);
        assert!(r.swizzle_enabled());
        assert_eq!(r.num_records(), 1000);
        assert_eq!(
            (r.dst_sel_x(), r.dst_sel_y(), r.dst_sel_z(), r.dst_sel_w()),
            (4, 5, 6, 7)
        );
        assert_eq!(r.dst_sel_xyzw(), 0xFAC);
        assert_eq!(r.dst_sel_xyz(), 0x1AC);
        assert_eq!(r.dst_sel_xy(), 0x2C);
        assert_eq!(r.nfmt(), 5);
        assert_eq!(r.dfmt(), 10);
        // Next-gen format overlays nfmt+dfmt: 5 | 10 << 3 = 0x55.
        assert_eq!(r.format(), 0x55);
        assert!(r.add_tid());
        assert_eq!(r.out_of_bounds(), 2);
        assert_eq!(r.base44(), 0x045_89AB_CDEF);
        assert_eq!(r.base48(), 0x0045_89AB_CDEF);
        // f3 bit 29 -> MemoryType bit 4 (Shader.h L591).
        assert_eq!(r.memory_type(), 0x10);
    }

    #[test]
    fn buffer_resource_update_address() {
        // Kyty: Shader.h UpdateAddress44 (L556) / UpdateAddress48 (L564).
        let mut r = ShaderBufferResource {
            fields: [0, 0x8123_0FFF, 0, 0],
        };
        r.update_address44(0x0ABC_1234_5678);
        assert_eq!(r.fields[0], 0x1234_5678);
        assert_eq!(r.fields[1], 0x8123_0ABC); // upper 20 bits kept
        assert_eq!(r.base44(), 0x0ABC_1234_5678);

        let mut r = ShaderBufferResource {
            fields: [0, 0x8123_FFFF, 0, 0],
        };
        r.update_address48(0xDEAD_1234_5678);
        assert_eq!(r.fields[0], 0x1234_5678);
        assert_eq!(r.fields[1], 0x8123_DEAD); // upper 16 bits kept
        assert_eq!(r.base48(), 0xDEAD_1234_5678);
    }

    #[test]
    fn texture_resource_legacy_accessors() {
        // Kyty: Shader.h ShaderTextureResource legacy fields (L655-672).
        // f1 = base_hi 0x06 | dfmt 0x0A << 20 | nfmt 9 << 26.
        // f2 = width 0xFF | height 0x100 << 14. f3 = type 9 << 28.
        // f4 = pitch 0x777 << 13.
        let r = ShaderTextureResource {
            fields: [
                0x1234_5678,
                0x24A0_0006,
                0x0040_00FF,
                0x9000_0000,
                0x00EE_E000,
                0,
                0,
                0,
            ],
        };
        assert_eq!(r.base38(), 0x612_3456_7800);
        assert_eq!(r.dfmt(), 0x0A);
        assert_eq!(r.nfmt(), 9);
        assert_eq!(r.width4(), 0xFF);
        assert_eq!(r.height4(), 0x100);
        assert_eq!(r.type_(), 9);
        assert_eq!(r.pitch(), 0x777);
        assert!(!r.pow2_pad());
        assert!(!r.interlaced());
    }

    #[test]
    fn texture_resource_next_gen_accessors() {
        // Kyty: Shader.h ShaderTextureResource next-gen fields (L631-653).
        // f1 = base_hi 0xAB | format 0x123 << 20. f2 = width5 low 0xFF.
        let r = ShaderTextureResource {
            fields: [0x1234_5678, 0x1230_00AB, 0x0040_00FF, 0, 0, 0, 0, 0],
        };
        assert_eq!(r.base40(), 0xAB12_3456_7800);
        assert_eq!(r.format(), 0x123);
        // width5 = (f1 >> 30 & 3) | (f2 & 0xFFF) << 2 = 0 | 0xFF << 2.
        assert_eq!(r.width5(), 0x3FC);
        assert_eq!(r.height5(), 0x100);
    }

    #[test]
    fn texture_resource_update_address() {
        // Kyty: Shader.h UpdateAddress38 (L601) / UpdateAddress40 (L609).
        let mut r = ShaderTextureResource {
            fields: [0, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        };
        r.update_address38(0x0000_0025_1234_5678);
        assert_eq!(r.fields[0], 0x1234_5678);
        assert_eq!(r.fields[1], 0xFFFF_FFE5); // low 6 bits replaced

        let mut r = ShaderTextureResource {
            fields: [0, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        };
        r.update_address40(0x0000_00A5_1234_5678);
        assert_eq!(r.fields[1], 0xFFFF_FFA5); // low 8 bits replaced
    }

    #[test]
    fn sampler_resource_accessors() {
        // Kyty: Shader.h ShaderSamplerResource (L675). Fields packed by hand.
        let s = ShaderSamplerResource {
            fields: [0xDC3E_D8D1, 0x8745_6123, 0x0795_5ABC, 0x9000_0ABC],
        };
        assert_eq!(s.clamp_x(), 1);
        assert_eq!(s.clamp_y(), 2);
        assert_eq!(s.clamp_z(), 3);
        assert_eq!(s.max_aniso_ratio(), 4);
        assert_eq!(s.depth_compare_func(), 5);
        assert!(s.force_unorm_coords());
        assert_eq!(s.aniso_threshold(), 6);
        assert!(s.mc_coord_trunc());
        assert!(s.force_degamma());
        assert_eq!(s.aniso_bias(), 0x21);
        assert!(s.trunc_coord());
        assert!(s.disable_cube_wrap());
        assert_eq!(s.filter_mode(), 2);
        assert!(s.skip_degamma());
        assert_eq!(s.min_lod(), 0x123);
        assert_eq!(s.max_lod(), 0x456);
        assert_eq!(s.perf_mip(), 7);
        assert_eq!(s.perf_z(), 8);
        assert_eq!(s.lod_bias(), 0x1ABC);
        assert_eq!(s.lod_bias_sec(), 0x15);
        assert_eq!(s.xy_mag_filter(), 1);
        assert_eq!(s.xy_min_filter(), 2);
        assert_eq!(s.z_filter(), 3);
        assert_eq!(s.mip_filter(), 1);
        assert_eq!(s.border_color_ptr(), 0xABC);
        assert_eq!(s.border_color_type(), 2);
        // The three bit-28 aliases (Shader.h L707-709).
        assert!(s.point_preclamp());
        assert!(s.aniso_override());
        assert!(s.blend_zero_prt());
    }

    #[test]
    fn gds_extended_and_index_resources() {
        // Kyty: Shader.h ShaderGdsResource (L714) / ShaderExtendedResource
        // (L727) / ShaderSamplerResource::UpdateIndex (L679).
        let gds = ShaderGdsResource { field: 0xDEAD_BEEF };
        assert_eq!(gds.base(), 0xDEAD);
        assert_eq!(gds.size(), 0xBEEF);

        let mut ext = ShaderExtendedResource::default();
        ext.update_address(0x1234_5678_9ABC_DEF0);
        assert_eq!(ext.fields, [0x9ABC_DEF0, 0x1234_5678]);
        assert_eq!(ext.base(), 0x1234_5678_9ABC_DEF0);

        let mut smp = ShaderSamplerResource::default();
        smp.update_index(42);
        assert_eq!(smp.fields[0], 42);
    }

    #[test]
    fn dst_sel_helpers() {
        // Kyty: Shader.h DstSel (L542) / GetDstSel (L547).
        assert_eq!(dst_sel(4, 5, 6, 7), 0xFAC);
        assert_eq!(get_dst_sel(0xFAC, 0), 4);
        assert_eq!(get_dst_sel(0xFAC, 1), 5);
        assert_eq!(get_dst_sel(0xFAC, 2), 6);
        assert_eq!(get_dst_sel(0xFAC, 3), 7);
    }

    #[test]
    fn shader_id_equality_on_all_three_fields() {
        // Kyty: Shader.h ShaderId::operator== (L538).
        let a = ShaderId {
            hash0: 1,
            crc32: 2,
            ids: vec![3, 4],
        };
        let mut b = a.clone();
        assert_eq!(a, b);
        b.ids.push(5);
        assert_ne!(a, b);
        let mut c = a.clone();
        c.hash0 = 9;
        assert_ne!(a, c);
        let mut d = a.clone();
        d.crc32 = 9;
        assert_ne!(a, d);
    }

    #[test]
    fn shader_sharp_bitfield() {
        // Kyty: Shader.h ShaderSharp (L905) -- offset_dw:15, size:1.
        let s = ShaderSharp::new(0x123, 0);
        assert_eq!((s.offset_dw(), s.size()), (0x123, 0));
        let s = ShaderSharp { raw: 0x8123 };
        assert_eq!((s.offset_dw(), s.size()), (0x123, 1));
        let sentinel = ShaderSharp::new(0x7fff, 0);
        assert_eq!(sentinel.offset_dw(), 0x7fff);
    }

    #[test]
    fn shader_semantic_bitfields() {
        // Kyty: Shader.h ShaderSemantic (L958). Hand-packed dword.
        let sem = ShaderSemantic { raw: 0x7565_3412 };
        assert_eq!(sem.semantic(), 0x12);
        assert_eq!(sem.hardware_mapping(), 0x34);
        assert_eq!(sem.size_in_elements(), 5);
        assert_eq!(sem.is_f16(), 2);
        assert!(sem.is_flat_shaded());
        assert!(!sem.is_linear());
        assert!(sem.is_custom());
        assert!(!sem.static_vb_index());
        assert!(sem.static_attribute());
        assert_eq!(sem.default_value(), 3);
        assert_eq!(sem.default_value_hi(), 1);
    }

    #[test]
    fn shader_draw_modifier_bitfields() {
        // Kyty: Shader.h ShaderDrawModifier (L927).
        let m = ShaderDrawModifier {
            raw: [0x169 | (0x55 << 9), 1],
        };
        assert!(m.enbl_start_vertex_offset());
        assert!(!m.enbl_start_index_offset());
        assert!(!m.enbl_start_instance_offset());
        assert!(m.enbl_draw_index());
        assert!(!m.enbl_user_vgprs());
        assert_eq!(m.render_target_slice_offset(), 0b011);
        assert!(m.fuse_draws());
        assert_eq!(m.compiler_flags(), 0x55);
        assert!(m.is_default());
    }
}
