//! Snapshot pins (insta) for the PM4/DCB fixture builders.
//!
//! These word streams are the M2/M3 acceptance inputs: every packet header,
//! register offset, and operand is load-bearing for the command processor.
//! A snapshot turns any accidental encoding change (a header refactor, a
//! register-constant edit) into a reviewable diff instead of a mysteriously
//! failing draw. Update intentionally with `cargo insta review` (or
//! `INSTA_UPDATE=always cargo test -p raeen-gpu --test pm4_snapshot`).

// The snapshot deliberately pins the deprecated M2 fixture builder's encoding
// for as long as the fixture exists; drop the allow when the fixture goes.
#[allow(deprecated)]
use raeen_gpu::{ScissorHalf, build_cp_draw_dcb, build_m2_draw_dcb};

/// Words formatted one-per-line in hex so snapshot diffs point at the exact
/// changed dword rather than one giant line.
fn hex_words(words: &[u32]) -> String {
    words
        .iter()
        .enumerate()
        .map(|(i, w)| format!("{i:04}: {w:#010x}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[allow(deprecated)] // pins the deprecated fixture's bytes until it is removed
fn m2_draw_dcb_encoding_is_pinned() {
    insta::assert_snapshot!(hex_words(&build_m2_draw_dcb()));
}

#[test]
fn cp_draw_dcb_scissor_halves_are_pinned() {
    insta::assert_snapshot!(hex_words(&build_cp_draw_dcb(64, 32, ScissorHalf::Left)));
    insta::assert_snapshot!(hex_words(&build_cp_draw_dcb(64, 32, ScissorHalf::Right)));
}
