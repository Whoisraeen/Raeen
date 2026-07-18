//! XPS5X-owned GPU boundary types.
//!
//! The HLE/runtime crates must not depend on Kyty's Rust data model. Conversion
//! into the port happens inside `xps5x-gpu`, allowing Kyty internals to evolve
//! without changing the emulator's public contracts.

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSharp {
    pub raw: u16,
}

impl ShaderSharp {
    #[must_use]
    pub const fn new(offset_dw: u16, size: u16) -> Self {
        Self {
            raw: (offset_dw & 0x7fff) | ((size & 1) << 15),
        }
    }

    #[must_use]
    pub const fn offset_dw(self) -> u16 {
        self.raw & 0x7fff
    }

    #[must_use]
    pub const fn size(self) -> u16 {
        self.raw >> 15
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderUserData {
    pub direct_resource_offset: Vec<u16>,
    pub sharp_resource_offset: [Vec<ShaderSharp>; 4],
    pub eud_size_dw: u16,
    pub srt_size_dw: u16,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSemantic {
    pub raw: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderMappedData {
    pub user_data: Option<ShaderUserData>,
    pub input_semantics: Vec<ShaderSemantic>,
}

impl From<ShaderSharp> for kyty_graphics::shader::ShaderSharp {
    fn from(value: ShaderSharp) -> Self {
        Self { raw: value.raw }
    }
}

impl From<ShaderUserData> for kyty_graphics::shader::ShaderUserData {
    fn from(value: ShaderUserData) -> Self {
        Self {
            direct_resource_offset: value.direct_resource_offset,
            sharp_resource_offset: value
                .sharp_resource_offset
                .map(|table| table.into_iter().map(Into::into).collect()),
            eud_size_dw: value.eud_size_dw,
            srt_size_dw: value.srt_size_dw,
        }
    }
}

impl From<ShaderSemantic> for kyty_graphics::shader::ShaderSemantic {
    fn from(value: ShaderSemantic) -> Self {
        Self { raw: value.raw }
    }
}

impl From<ShaderMappedData> for kyty_graphics::shader::ShaderMappedData {
    fn from(value: ShaderMappedData) -> Self {
        Self {
            user_data: value.user_data.map(Into::into),
            input_semantics: value.input_semantics.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xps5x_shader_contract_converts_at_the_kyty_boundary() {
        let data = ShaderMappedData {
            user_data: Some(ShaderUserData {
                direct_resource_offset: vec![3],
                sharp_resource_offset: [vec![ShaderSharp::new(7, 1)], vec![], vec![], vec![]],
                eud_size_dw: 8,
                srt_size_dw: 16,
            }),
            input_semantics: vec![ShaderSemantic { raw: 0x1234 }],
        };
        let kyty: kyty_graphics::shader::ShaderMappedData = data.into();
        assert_eq!(
            kyty.user_data.unwrap().sharp_resource_offset[0][0].raw,
            0x8007
        );
        assert_eq!(kyty.input_semantics[0].raw, 0x1234);
    }
}
