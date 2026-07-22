//! Raeen-owned GPU boundary types.
//!
//! The HLE/runtime crates must not depend on Kyty's Rust data model. Conversion
//! into the port happens inside `raeen-gpu`, allowing Kyty internals to evolve
//! without changing the emulator's public contracts.

pub use raeen_core::subsystems::{ShaderMappedData, ShaderSemantic, ShaderSharp, ShaderUserData};

fn sharp_to_kyty(value: ShaderSharp) -> kyty_graphics::shader::ShaderSharp {
    kyty_graphics::shader::ShaderSharp { raw: value.raw }
}

fn user_data_to_kyty(value: ShaderUserData) -> kyty_graphics::shader::ShaderUserData {
    kyty_graphics::shader::ShaderUserData {
        direct_resource_offset: value.direct_resource_offset,
        sharp_resource_offset: value
            .sharp_resource_offset
            .map(|table| table.into_iter().map(sharp_to_kyty).collect()),
        eud_size_dw: value.eud_size_dw,
        srt_size_dw: value.srt_size_dw,
    }
}

fn semantic_to_kyty(value: ShaderSemantic) -> kyty_graphics::shader::ShaderSemantic {
    kyty_graphics::shader::ShaderSemantic { raw: value.raw }
}

pub(crate) fn mapped_data_to_kyty(
    value: ShaderMappedData,
) -> kyty_graphics::shader::ShaderMappedData {
    kyty_graphics::shader::ShaderMappedData {
        user_data: value.user_data.map(user_data_to_kyty),
        input_semantics: value
            .input_semantics
            .into_iter()
            .map(semantic_to_kyty)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raeen_shader_contract_converts_at_the_kyty_boundary() {
        let data = ShaderMappedData {
            user_data: Some(ShaderUserData {
                direct_resource_offset: vec![3],
                sharp_resource_offset: [vec![ShaderSharp::new(7, 1)], vec![], vec![], vec![]],
                eud_size_dw: 8,
                srt_size_dw: 16,
            }),
            input_semantics: vec![ShaderSemantic { raw: 0x1234 }],
        };
        let kyty = mapped_data_to_kyty(data);
        assert_eq!(
            kyty.user_data.unwrap().sharp_resource_offset[0][0].raw,
            0x8007
        );
        assert_eq!(kyty.input_semantics[0].raw, 0x1234);
    }
}
