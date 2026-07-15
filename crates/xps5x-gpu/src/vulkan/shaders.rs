//! Hand-built SPIR-V for the offscreen triangle.
//!
//! # Why this is hand-built, and what replaces it
//!
//! XPS5X's real shader path is `shader::gcn_decoder` → `shader::ir` →
//! [`shader::spirv_emitter`](crate::shader::spirv_emitter), which translates a
//! title's RDNA2 ISA into SPIR-V.
//!
//! That path's **I/O model is now real**: `ExportPosition` emits a `vec4`
//! decorated `BuiltIn Position`, `ExportColor` a `vec4` at a Location, and both
//! compose their components (padding a short export to `w = 1`). So the shape a
//! driver requires is there — see `spirv_emitter`'s
//! `position_export_is_a_vec4_builtin_position_not_a_location`.
//!
//! What is still missing before this module can be deleted is the **body**, not
//! the interface: `gcn_decoder` must lower enough real RDNA2 ISA that a title's
//! actual vertex/fragment pair round-trips. Until a decoded shader is proven to
//! draw, swapping these hand-built modules out would trade a known-good input
//! for an unknown one and make any failure ambiguous.
//!
//! Rather than fake that, this module emits two minimal, purpose-built shaders
//! directly. They exist to prove the **Vulkan half** of the pipeline — device,
//! pipeline, draw, readback — with a known-good input, so that when the
//! GCN→SPIR-V translation lands it plugs into a backend already verified to
//! draw. Hooking `spirv_emitter` up here is the next step, not a done one.
//!
//! The modules are built with a tiny instruction writer instead of being pasted
//! in as an opaque `[u32]` blob, so every opcode and id is reviewable.

// ─── SPIR-V binary layout ──────────────────────────────────
const SPIRV_MAGIC: u32 = 0x0723_0203;
/// SPIR-V 1.0 — the baseline every Vulkan 1.0+ driver accepts. Nothing here
/// needs a later version.
const SPIRV_VERSION: u32 = 0x0001_0000;
/// Generator magic number. 0 = "unknown / not registered with Khronos", which
/// is the honest value for an in-house emitter.
const SPIRV_GENERATOR: u32 = 0;

// ─── Opcodes ───────────────────────────────────────────────
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_COMPOSITE: u16 = 44;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_DECORATE: u16 = 71;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;

// ─── Enum operands ─────────────────────────────────────────
const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;
const EXECUTION_MODEL_VERTEX: u32 = 0;
const EXECUTION_MODEL_FRAGMENT: u32 = 4;
const EXECUTION_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_OUTPUT: u32 = 3;
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_LOCATION: u32 = 30;
const BUILTIN_POSITION: u32 = 0;
const FUNCTION_CONTROL_NONE: u32 = 0;

/// Accumulates SPIR-V words.
struct Words(Vec<u32>);

impl Words {
    /// Start a module: 5-word header with a caller-supplied id bound.
    fn with_header(id_bound: u32) -> Self {
        Self(vec![
            SPIRV_MAGIC,
            SPIRV_VERSION,
            SPIRV_GENERATOR,
            id_bound,
            0, // reserved schema
        ])
    }

    /// Append one instruction: `word_count << 16 | opcode`, then operands.
    fn op(&mut self, opcode: u16, operands: &[u32]) {
        let word_count = u32::try_from(operands.len() + 1).expect("instruction operand overflow");
        self.0.push((word_count << 16) | u32::from(opcode));
        self.0.extend_from_slice(operands);
    }

    /// Append an instruction whose trailing operand is a literal string.
    fn op_str(&mut self, opcode: u16, leading: &[u32], text: &str, trailing: &[u32]) {
        let mut operands = leading.to_vec();
        operands.extend_from_slice(&literal_string(text));
        operands.extend_from_slice(trailing);
        self.op(opcode, &operands);
    }

    fn finish(self) -> Vec<u32> {
        self.0
    }
}

/// Encode a SPIR-V literal string: UTF-8, NUL-terminated, zero-padded to a
/// whole number of little-endian words.
fn literal_string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0); // NUL terminator
    bytes.resize(bytes.len().next_multiple_of(4), 0); // zero-pad to whole words
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Vertex shader: pass a `vec4` vertex attribute straight through to `Position`.
///
/// GLSL equivalent:
/// ```glsl
/// layout(location = 0) in vec4 inPos;
/// void main() { gl_Position = inPos; }
/// ```
pub fn triangle_vertex_spirv() -> Vec<u32> {
    // Ids, assigned up front so the layout below reads like the GLSL above.
    const MAIN: u32 = 1;
    const VOID: u32 = 2;
    const FN_TYPE: u32 = 3;
    const FLOAT: u32 = 4;
    const V4FLOAT: u32 = 5;
    const PTR_IN_V4: u32 = 6;
    const PTR_OUT_V4: u32 = 7;
    const IN_POS: u32 = 8;
    const OUT_POSITION: u32 = 9;
    const LABEL: u32 = 10;
    const LOADED: u32 = 11;
    const ID_BOUND: u32 = 12;

    let mut w = Words::with_header(ID_BOUND);

    // Section 1-2: capabilities, memory model.
    w.op(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    w.op(
        OP_MEMORY_MODEL,
        &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
    );

    // Section 3: entry point. The interface list must name every Input/Output
    // variable the entry point statically uses.
    w.op_str(
        OP_ENTRY_POINT,
        &[EXECUTION_MODEL_VERTEX, MAIN],
        "main",
        &[IN_POS, OUT_POSITION],
    );
    // Section 4: no execution modes are required for a vertex shader.

    // Section 5: decorations.
    w.op(
        OP_DECORATE,
        &[OUT_POSITION, DECORATION_BUILTIN, BUILTIN_POSITION],
    );
    w.op(OP_DECORATE, &[IN_POS, DECORATION_LOCATION, 0]);

    // Section 6: types, then global variables.
    w.op(OP_TYPE_VOID, &[VOID]);
    w.op(OP_TYPE_FUNCTION, &[FN_TYPE, VOID]);
    w.op(OP_TYPE_FLOAT, &[FLOAT, 32]);
    w.op(OP_TYPE_VECTOR, &[V4FLOAT, FLOAT, 4]);
    w.op(OP_TYPE_POINTER, &[PTR_IN_V4, STORAGE_CLASS_INPUT, V4FLOAT]);
    w.op(
        OP_TYPE_POINTER,
        &[PTR_OUT_V4, STORAGE_CLASS_OUTPUT, V4FLOAT],
    );
    w.op(OP_VARIABLE, &[PTR_IN_V4, IN_POS, STORAGE_CLASS_INPUT]);
    w.op(
        OP_VARIABLE,
        &[PTR_OUT_V4, OUT_POSITION, STORAGE_CLASS_OUTPUT],
    );

    // Section 7: the function body.
    w.op(OP_FUNCTION, &[VOID, MAIN, FUNCTION_CONTROL_NONE, FN_TYPE]);
    w.op(OP_LABEL, &[LABEL]);
    w.op(OP_LOAD, &[V4FLOAT, LOADED, IN_POS]);
    w.op(OP_STORE, &[OUT_POSITION, LOADED]);
    w.op(OP_RETURN, &[]);
    w.op(OP_FUNCTION_END, &[]);

    w.finish()
}

/// The color the fragment shader writes, as linear RGBA.
///
/// Kept public so the acceptance test asserts against the same constant the
/// shader actually emits, rather than a copy that could drift.
pub const TRIANGLE_COLOR: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

/// Fragment shader: write [`TRIANGLE_COLOR`] to color attachment 0.
///
/// GLSL equivalent:
/// ```glsl
/// layout(location = 0) out vec4 outColor;
/// void main() { outColor = vec4(0.0, 1.0, 0.0, 1.0); }
/// ```
pub fn triangle_fragment_spirv() -> Vec<u32> {
    const MAIN: u32 = 1;
    const VOID: u32 = 2;
    const FN_TYPE: u32 = 3;
    const FLOAT: u32 = 4;
    const V4FLOAT: u32 = 5;
    const PTR_OUT_V4: u32 = 6;
    const OUT_COLOR: u32 = 7;
    const CONST_ZERO: u32 = 8;
    const CONST_ONE: u32 = 9;
    const COLOR: u32 = 10;
    const LABEL: u32 = 11;
    const ID_BOUND: u32 = 12;

    let mut w = Words::with_header(ID_BOUND);

    w.op(OP_CAPABILITY, &[CAPABILITY_SHADER]);
    w.op(
        OP_MEMORY_MODEL,
        &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
    );
    w.op_str(
        OP_ENTRY_POINT,
        &[EXECUTION_MODEL_FRAGMENT, MAIN],
        "main",
        &[OUT_COLOR],
    );
    // Vulkan requires fragment shaders to declare their framebuffer origin.
    w.op(OP_EXECUTION_MODE, &[MAIN, EXECUTION_MODE_ORIGIN_UPPER_LEFT]);
    w.op(OP_DECORATE, &[OUT_COLOR, DECORATION_LOCATION, 0]);

    w.op(OP_TYPE_VOID, &[VOID]);
    w.op(OP_TYPE_FUNCTION, &[FN_TYPE, VOID]);
    w.op(OP_TYPE_FLOAT, &[FLOAT, 32]);
    w.op(OP_TYPE_VECTOR, &[V4FLOAT, FLOAT, 4]);
    w.op(
        OP_TYPE_POINTER,
        &[PTR_OUT_V4, STORAGE_CLASS_OUTPUT, V4FLOAT],
    );
    w.op(OP_VARIABLE, &[PTR_OUT_V4, OUT_COLOR, STORAGE_CLASS_OUTPUT]);

    // SPIR-V forbids duplicate constants of the same type and value, so 0.0 and
    // 1.0 are each declared once and reused by the composite below. This is why
    // `TRIANGLE_COLOR` is restricted to those two values.
    w.op(OP_CONSTANT, &[FLOAT, CONST_ZERO, 0.0f32.to_bits()]);
    w.op(OP_CONSTANT, &[FLOAT, CONST_ONE, 1.0f32.to_bits()]);
    let component = |v: f32| if v == 0.0 { CONST_ZERO } else { CONST_ONE };
    w.op(
        OP_CONSTANT_COMPOSITE,
        &[
            V4FLOAT,
            COLOR,
            component(TRIANGLE_COLOR[0]),
            component(TRIANGLE_COLOR[1]),
            component(TRIANGLE_COLOR[2]),
            component(TRIANGLE_COLOR[3]),
        ],
    );

    w.op(OP_FUNCTION, &[VOID, MAIN, FUNCTION_CONTROL_NONE, FN_TYPE]);
    w.op(OP_LABEL, &[LABEL]);
    w.op(OP_STORE, &[OUT_COLOR, COLOR]);
    w.op(OP_RETURN, &[]);
    w.op(OP_FUNCTION_END, &[]);

    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::binary::Disassemble;

    /// `TRIANGLE_COLOR` may only contain 0.0/1.0 — see the constant-dedup note
    /// in `triangle_fragment_spirv`. If someone picks a new color, this fails
    /// loudly instead of silently emitting a wrong composite.
    #[test]
    fn triangle_color_uses_only_declared_constants() {
        for c in TRIANGLE_COLOR {
            assert!(
                c == 0.0 || c == 1.0,
                "TRIANGLE_COLOR component {c} needs a new OpConstant in triangle_fragment_spirv"
            );
        }
    }

    #[test]
    fn literal_string_is_nul_terminated_and_padded() {
        // "main" is 4 bytes + NUL -> padded to 8 bytes = 2 words.
        assert_eq!(literal_string("main"), vec![0x6e69_616d, 0x0000_0000]);
        // "ab" is 2 bytes + NUL -> padded to 4 bytes = 1 word.
        assert_eq!(literal_string("ab"), vec![0x0000_6261]);
    }

    fn parse(words: &[u32]) -> rspirv::dr::Module {
        rspirv::dr::load_words(words).expect("emitted SPIR-V must parse")
    }

    #[test]
    fn vertex_module_is_structurally_valid() {
        let words = triangle_vertex_spirv();
        assert_eq!(words[0], SPIRV_MAGIC);
        let module = parse(&words);

        let entry = module
            .entry_points
            .first()
            .expect("vertex module declares an entry point");
        assert_eq!(
            entry.operands[0].unwrap_execution_model(),
            rspirv::spirv::ExecutionModel::Vertex
        );

        // The pass-through must actually load the input and store to Position.
        let text = module.disassemble();
        assert!(
            text.contains("OpLoad"),
            "vertex body loads the attribute:\n{text}"
        );
        assert!(
            text.contains("OpStore"),
            "vertex body stores Position:\n{text}"
        );
        assert!(
            text.contains("BuiltIn Position"),
            "output must be decorated as Position:\n{text}"
        );
    }

    #[test]
    fn fragment_module_is_structurally_valid() {
        let words = triangle_fragment_spirv();
        assert_eq!(words[0], SPIRV_MAGIC);
        let module = parse(&words);

        let entry = module
            .entry_points
            .first()
            .expect("fragment module declares an entry point");
        assert_eq!(
            entry.operands[0].unwrap_execution_model(),
            rspirv::spirv::ExecutionModel::Fragment
        );

        let text = module.disassemble();
        assert!(
            text.contains("OriginUpperLeft"),
            "Vulkan requires an origin execution mode:\n{text}"
        );
        assert!(
            text.contains("OpConstantComposite"),
            "color must be a composite constant:\n{text}"
        );
    }

    /// The id bound in the header must exceed every id actually used, or
    /// drivers reject the module.
    #[test]
    fn id_bounds_cover_all_used_ids() {
        for words in [triangle_vertex_spirv(), triangle_fragment_spirv()] {
            let bound = words[3];
            let module = parse(&words);
            let max_used = module
                .all_inst_iter()
                .filter_map(|i| i.result_id)
                .max()
                .expect("module defines at least one id");
            assert!(
                max_used < bound,
                "id bound {bound} must exceed max used id {max_used}"
            );
        }
    }
}
