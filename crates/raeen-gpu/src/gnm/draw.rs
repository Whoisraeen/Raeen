//! Draw call translation — GNM draw commands → Vulkan draw calls.

use tracing::debug;

/// Translate a GNM indexed draw call to Vulkan parameters.
#[derive(Debug, Clone)]
pub struct TranslatedDraw {
    /// Number of indices.
    pub index_count: u32,
    /// Number of instances.
    pub instance_count: u32,
    /// First index offset.
    pub first_index: u32,
    /// Vertex offset added to each index.
    pub vertex_offset: i32,
    /// First instance ID.
    pub first_instance: u32,
    /// Index buffer GPU address.
    pub index_buffer_addr: u64,
    /// Index type (16-bit or 32-bit).
    pub index_type: IndexType,
}

/// Index buffer element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    U16,
    U32,
}

/// Translate a PM4 DRAW_INDEX_2 packet.
pub fn translate_draw_index_2(body: &[u32]) -> Option<TranslatedDraw> {
    if body.len() < 4 {
        return None;
    }

    let max_size = body[0];
    let index_base_lo = body[1];
    let index_base_hi = body[2];
    let index_count_and_draw_initiator = body[3];
    let index_count = index_count_and_draw_initiator;

    let index_buffer_addr = ((index_base_hi as u64) << 32) | (index_base_lo as u64);

    debug!(
        "DRAW_INDEX_2: count={}, ib_addr={:#x}, max={}",
        index_count, index_buffer_addr, max_size
    );

    Some(TranslatedDraw {
        index_count,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
        index_buffer_addr,
        index_type: IndexType::U32, // Default; actual type from register state.
    })
}

/// Translate a PM4 DRAW_INDEX_AUTO packet (non-indexed draw).
pub fn translate_draw_auto(body: &[u32]) -> Option<TranslatedDraw> {
    if body.is_empty() {
        return None;
    }

    let vertex_count = body[0];

    debug!("DRAW_INDEX_AUTO: vertex_count={}", vertex_count);

    Some(TranslatedDraw {
        index_count: vertex_count,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
        index_buffer_addr: 0,
        index_type: IndexType::U32,
    })
}
