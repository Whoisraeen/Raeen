//! Whether the draws that reach the Vulkan backend can rasterize anything.
//!
//! The draw counters that already exist (`decoded_draws` / `executed_draws`,
//! see [`crate::agc_exec`]) answer "did the packet reach the sink". They cannot
//! answer the next question, which is the one a black frame actually poses:
//! **could that draw have covered a pixel at all?** A draw with a zero vertex
//! count, a draw whose every index is the same value, and a vertex program that
//! never exports a clip position all reach the sink, all submit successfully,
//! all report zero validation errors, and all leave the render target holding
//! exactly what the attachment LOAD/CLEAR put there.
//!
//! Those three are structurally distinguishable *before* the GPU runs, from
//! data the draw path already computed, so this census is always on and costs a
//! handful of relaxed `fetch_add`s per draw — no env var, no allocation, no
//! extra pass over any buffer. `RAEEN_PIPELINE_STATS` (see
//! [`crate::vulkan::offscreen`]) measures what the *hardware* then did with the
//! survivors; this measures what was handed to it.
//!
//! Reported alongside `CHAIN CENSUS` so a single log carries the whole
//! draw-accounting chain: decoded → executed → could-rasterize → rasterized.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Draws that reached [`crate::vulkan::offscreen::render_draw`] or its deferred
/// twin — the same population `executed_draws` counts.
static DRAWS: AtomicU64 = AtomicU64::new(0);
/// Draws whose vertex/index count is 0. `vkCmdDraw(count = 0)` is legal and
/// draws nothing, so these are silent no-ops today.
static ZERO_COUNT: AtomicU64 = AtomicU64::new(0);
/// Draws carrying fewer vertices than one primitive of their topology needs
/// (a 2-vertex triangle list, a 1-vertex line list). Also silent no-ops.
static BELOW_PRIMITIVE_MINIMUM: AtomicU64 = AtomicU64::new(0);
/// Indexed draws.
static INDEXED: AtomicU64 = AtomicU64::new(0);
/// Indexed draws whose largest index is 0 — every vertex of every primitive is
/// the same record, so every primitive has zero area.
static INDEXED_DEGENERATE: AtomicU64 = AtomicU64::new(0);
/// Vertex programs handed to the recompiler (cache misses only).
static VS_TRANSLATED: AtomicU64 = AtomicU64::new(0);
/// Vertex programs with no `exp pos0` anywhere in the ISA. Their `gl_Position`
/// is never stored, so every vertex sits at the undefined initial value and no
/// primitive survives clipping — see
/// [`kyty_graphics::shader::shader_exports_position`].
static VS_WITHOUT_POSITION_EXPORT: AtomicU64 = AtomicU64::new(0);
/// Distinct graphics stage interfaces whose EMITTED vertex SPIR-V was scanned
/// for a `gl_Position` store, and how many had none. This is the other half of
/// the same question: the ISA can carry `exp pos0` while the recompiled module
/// does not store it.
static VS_SPIRV_SCANNED: AtomicU64 = AtomicU64::new(0);
static VS_SPIRV_WITHOUT_POSITION_STORE: AtomicU64 = AtomicU64::new(0);

/// One census reading. Cumulative since process start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrawGeometryCensus {
    pub draws: u64,
    pub zero_count: u64,
    pub below_primitive_minimum: u64,
    pub indexed: u64,
    pub indexed_degenerate: u64,
    pub vs_translated: u64,
    pub vs_without_position_export: u64,
    pub vs_spirv_scanned: u64,
    pub vs_spirv_without_position_store: u64,
}

impl DrawGeometryCensus {
    /// Draws that cannot have rasterized a primitive for a reason visible on
    /// the CPU side. Not a total of the fields: a zero-count draw is also
    /// below its topology's minimum, so the two must not be added.
    #[must_use]
    pub const fn cannot_rasterize(&self) -> u64 {
        self.below_primitive_minimum + self.indexed_degenerate
    }
}

/// Vertices one primitive of `prim_type` needs, or `None` for a primitive type
/// the draw path does not map (those are refused upstream by name).
///
/// Strips/fans need the same minimum as their list form for the FIRST
/// primitive; that is exactly the threshold below which nothing rasterizes.
const fn vertices_per_primitive(prim_type: u32) -> Option<u32> {
    Some(match prim_type {
        // NONE draws nothing on hardware and is consumed before this point.
        0 => return None,
        1 => 1,     // point list
        2 | 3 => 2, // line list / strip
        4..=7 => 3, // triangle list / fan / strip / polygon
        17 => 3,    // rect list, rasterized as a strip quad
        _ => return None,
    })
}

/// Record one draw that reached the backend.
///
/// `max_index` is the largest value in the bound index buffer for an indexed
/// draw (the draw path already computes it to size the vertex upload — see
/// `required_vertex_records`), or `None` for a non-indexed draw.
pub(crate) fn note_draw(prim_type: u32, count: u32, max_index: Option<u32>) {
    DRAWS.fetch_add(1, Relaxed);
    if count == 0 {
        ZERO_COUNT.fetch_add(1, Relaxed);
    }
    if let Some(minimum) = vertices_per_primitive(prim_type)
        && count < minimum
    {
        BELOW_PRIMITIVE_MINIMUM.fetch_add(1, Relaxed);
    }
    if let Some(max_index) = max_index {
        INDEXED.fetch_add(1, Relaxed);
        // A single-vertex draw legitimately has max_index 0; only a draw that
        // claims a whole primitive out of one repeated record is degenerate.
        if max_index == 0 && count > 1 {
            INDEXED_DEGENERATE.fetch_add(1, Relaxed);
        }
    }
}

/// Record one vertex-program recompilation and whether it exports a clip
/// position. Called on cache misses only, so this is per distinct shader.
pub(crate) fn note_vertex_translation(exports_position: bool) {
    VS_TRANSLATED.fetch_add(1, Relaxed);
    if !exports_position {
        VS_WITHOUT_POSITION_EXPORT.fetch_add(1, Relaxed);
    }
}

/// Record whether one distinct emitted vertex module stores `gl_Position`.
/// Called once per (vs, fs, topology) interface, not per draw.
pub(crate) fn note_vertex_position_store(stores_position: bool) {
    VS_SPIRV_SCANNED.fetch_add(1, Relaxed);
    if !stores_position {
        VS_SPIRV_WITHOUT_POSITION_STORE.fetch_add(1, Relaxed);
    }
}

/// Read the census.
#[must_use]
pub fn snapshot() -> DrawGeometryCensus {
    DrawGeometryCensus {
        draws: DRAWS.load(Relaxed),
        zero_count: ZERO_COUNT.load(Relaxed),
        below_primitive_minimum: BELOW_PRIMITIVE_MINIMUM.load(Relaxed),
        indexed: INDEXED.load(Relaxed),
        indexed_degenerate: INDEXED_DEGENERATE.load(Relaxed),
        vs_translated: VS_TRANSLATED.load(Relaxed),
        vs_without_position_export: VS_WITHOUT_POSITION_EXPORT.load(Relaxed),
        vs_spirv_scanned: VS_SPIRV_SCANNED.load(Relaxed),
        vs_spirv_without_position_store: VS_SPIRV_WITHOUT_POSITION_STORE.load(Relaxed),
    }
}

/// Emit one census line. Called on the `CHAIN CENSUS` cadence.
pub(crate) fn report() {
    let c = snapshot();
    if c.draws == 0 && c.vs_translated == 0 {
        return;
    }
    tracing::info!(
        draws = c.draws,
        zero_count = c.zero_count,
        below_primitive_minimum = c.below_primitive_minimum,
        indexed = c.indexed,
        indexed_degenerate = c.indexed_degenerate,
        cannot_rasterize = c.cannot_rasterize(),
        vs_translated = c.vs_translated,
        vs_without_position_export = c.vs_without_position_export,
        vs_spirv_scanned = c.vs_spirv_scanned,
        vs_spirv_without_position_store = c.vs_spirv_without_position_store,
        "DRAW GEOMETRY CENSUS: of the draws that reached the Vulkan backend, how many could have \
         covered a pixel at all. `zero_count` and `below_primitive_minimum` are silent no-ops \
         (vkCmdDraw with too few vertices is legal and draws nothing); `indexed_degenerate` is an \
         index buffer whose largest value is 0, collapsing every primitive onto one record; \
         `vs_without_position_export` counts distinct vertex programs with no `exp pos0` in the \
         ISA and `vs_spirv_without_position_store` distinct EMITTED modules with no store to \
         gl_Position — either way gl_Position keeps its undefined all-zero value, every vertex \
         has w == 0, and no primitive survives clipping. A nonzero column here is a BLACK-FRAME \
         cause, not a warning (cumulative; always on)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-global, so the unit tests assert on deltas and
    /// on the pure classification helper rather than on absolute values.
    #[test]
    fn topology_minimums_match_the_mapped_primitive_types() {
        assert_eq!(vertices_per_primitive(0), None, "NONE draws nothing");
        assert_eq!(vertices_per_primitive(1), Some(1), "point list");
        assert_eq!(vertices_per_primitive(2), Some(2), "line list");
        assert_eq!(vertices_per_primitive(3), Some(2), "line strip");
        for triangle in [4, 5, 6, 7] {
            assert_eq!(
                vertices_per_primitive(triangle),
                Some(3),
                "primitive type {triangle} needs three vertices"
            );
        }
        assert_eq!(vertices_per_primitive(17), Some(3), "rect list");
        assert_eq!(
            vertices_per_primitive(9),
            None,
            "an unmapped primitive type must not be scored"
        );
    }

    #[test]
    fn a_two_vertex_triangle_and_a_zero_count_draw_are_both_counted() {
        let before = snapshot();
        note_draw(4, 0, None);
        note_draw(4, 2, None);
        note_draw(4, 3, None);
        let after = snapshot();
        assert_eq!(after.draws - before.draws, 3);
        assert_eq!(after.zero_count - before.zero_count, 1, "only the count==0");
        assert_eq!(
            after.below_primitive_minimum - before.below_primitive_minimum,
            2,
            "count 0 and count 2 are both below a triangle's three vertices"
        );
    }

    #[test]
    fn an_all_zero_index_buffer_is_degenerate_but_a_single_index_is_not() {
        let before = snapshot();
        note_draw(4, 6, Some(0));
        note_draw(1, 1, Some(0));
        note_draw(4, 6, Some(5));
        let after = snapshot();
        assert_eq!(after.indexed - before.indexed, 3);
        assert_eq!(
            after.indexed_degenerate - before.indexed_degenerate,
            1,
            "only the 6-index draw whose largest index is 0"
        );
    }

    #[test]
    fn a_vertex_program_without_a_position_export_is_counted() {
        let before = snapshot();
        note_vertex_translation(true);
        note_vertex_translation(false);
        let after = snapshot();
        assert_eq!(after.vs_translated - before.vs_translated, 2);
        assert_eq!(
            after.vs_without_position_export - before.vs_without_position_export,
            1
        );
    }

    /// The ISA-side and SPIR-V-side position counters are separate on purpose:
    /// an `exp pos0` the recompiler drops shows up only in the second.
    #[test]
    fn the_emitted_module_position_store_is_counted_separately_from_the_isa_export() {
        let before = snapshot();
        note_vertex_position_store(true);
        note_vertex_position_store(false);
        let after = snapshot();
        assert_eq!(after.vs_spirv_scanned - before.vs_spirv_scanned, 2);
        assert_eq!(
            after.vs_spirv_without_position_store - before.vs_spirv_without_position_store,
            1
        );
        assert_eq!(
            after.vs_translated, before.vs_translated,
            "the SPIR-V scan must not touch the ISA-export counters"
        );
    }
}
