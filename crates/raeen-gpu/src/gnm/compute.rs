//! Compute dispatch translation.

use tracing::debug;

/// Translated compute dispatch parameters.
#[derive(Debug, Clone)]
pub struct TranslatedDispatch {
    /// Thread groups in X dimension.
    pub group_count_x: u32,
    /// Thread groups in Y dimension.
    pub group_count_y: u32,
    /// Thread groups in Z dimension.
    pub group_count_z: u32,
}

/// Translate a PM4 DISPATCH_DIRECT packet.
pub fn translate_dispatch_direct(body: &[u32]) -> Option<TranslatedDispatch> {
    if body.len() < 3 {
        return None;
    }

    let dispatch = TranslatedDispatch {
        group_count_x: body[0],
        group_count_y: body[1],
        group_count_z: body[2],
    };

    debug!(
        "DISPATCH_DIRECT: groups=({}, {}, {})",
        dispatch.group_count_x, dispatch.group_count_y, dispatch.group_count_z
    );

    Some(dispatch)
}
