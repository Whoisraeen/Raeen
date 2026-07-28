//! Ordered GPU side effects, session level (checklist item 5, steps 4–5):
//! a state-only DCB carrying an `IT_EVENT_WRITE` and an embedded AGC flip is
//! executed through the real `CommandProcessor`, and the recorded side
//! effects reach the process-global hand-off queue **iff** the
//! `RAEEN_DEFER_GPU_SIDE_EFFECTS` gate is on — with the gate off (the
//! default) the queue stays empty, because the HLE already applied the
//! eager duplicates at submit time and a second delivery would double-flip.
//!
//! Runs without Vulkan: a draw-free DCB takes the state-only path.

use raeen_gpu::GpuGuestMemory;
use raeen_gpu::agc_exec::AgcGpuSession;
use raeen_gpu::ordered_side_effects::{OrderedGpuSideEffect, drain};
use std::sync::Arc;

/// No packet in the fixture DCB touches guest memory.
struct NoMemory;

impl GpuGuestMemory for NoMemory {
    fn validate_gpu_range(&self, _addr: u64, _len: u64, _write: bool) -> bool {
        false
    }
    fn read_gpu(&self, _addr: u64, _out: &mut [u8]) -> bool {
        false
    }
    fn write_gpu(&self, _addr: u64, _data: &[u8]) -> bool {
        false
    }
}

/// State-only DCB: standard `IT_EVENT_WRITE` (event id 0x2A), then an AGC
/// flip packet (`IT_NOP` + `R_FLIP`) — handle 1, buffer 2, mode 1, arg 9.
fn side_effect_dcb() -> Vec<u32> {
    use kyty_graphics::pm4;
    vec![
        pm4::header(3, pm4::IT_EVENT_WRITE, pm4::R_ZERO),
        0x2A,
        0,
        pm4::header(7, pm4::IT_NOP, pm4::R_FLIP),
        1, // video out handle
        2, // display buffer index
        1, // flip mode
        9, // flip arg lo
        0, // flip arg hi
        0,
    ]
}

struct EnvReset;
impl Drop for EnvReset {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("RAEEN_DEFER_GPU_SIDE_EFFECTS") };
    }
}

/// One test, both policies in sequence (the gate and the queue are process
/// state; a single test needs no cross-test lock).
#[test]
fn cp_executed_side_effects_reach_the_queue_only_under_the_gate() {
    let _reset = EnvReset;
    let session = AgcGpuSession::new_process(Arc::new(NoMemory));
    let dcb = side_effect_dcb();

    // Gate OFF (default): the worker records the effects but publishes
    // nothing — the eager submit-time duplicates own delivery.
    unsafe { std::env::remove_var("RAEEN_DEFER_GPU_SIDE_EFFECTS") };
    let _ = drain();
    session
        .execute_dcb_cp(&dcb, false)
        .expect("a state-only DCB executes without Vulkan");
    assert!(
        drain().is_empty(),
        "gate off: the worker must not double-deliver the eager side effects"
    );

    // Gate ON: the worker's in-stream execution is the only delivery source,
    // in PM4 stream order.
    unsafe { std::env::set_var("RAEEN_DEFER_GPU_SIDE_EFFECTS", "1") };
    session
        .execute_dcb_cp(&dcb, false)
        .expect("a state-only DCB executes without Vulkan");
    assert_eq!(
        drain(),
        vec![
            OrderedGpuSideEffect::EventWrite { event_id: 0x2A },
            OrderedGpuSideEffect::Flip {
                video_out_handle: 1,
                display_buffer_index: 2,
                flip_mode: 1,
                flip_arg: 9,
            },
        ],
        "gate on: the CP-executed effects reach the hand-off queue in order"
    );
    session.shutdown();
}
