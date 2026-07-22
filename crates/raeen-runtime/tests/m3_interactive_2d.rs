//! M3 acceptance test: an interactive 2D homebrew, end to end.
//!
//! A fully synthetic homebrew module — hand-assembled x86-64 with two import
//! slots — runs through the **real** runtime ([`execute_linked`], the same
//! GuestArena + VEH trap-and-dispatch path titles use) and:
//!
//!   1. calls the real `libScePad::scePadReadState(handle=1, &pad)`,
//!   2. loads the `buttons` u32 from the returned `ScePadData`,
//!   3. CPU-draws a 4x4 linear RGBA8 display buffer in guest memory —
//!      white (`0xFFFFFFFF`) when any button is held, black (`0xFF000000`)
//!      otherwise,
//!   4. calls the real `libSceVideoOut::sceVideoOutSubmitFlip(handle=1, 0, ..)`,
//!   5. reads pixel 0 back into `RAX` and returns.
//!
//! The buffer is registered with VideoOut (linear RGBA8) so the flip routes it
//! through the present-from-guest-memory path (SharpEmu
//! `VulkanVideoPresenter.cs:1643-1660` `GuestImageWantsInitialData`): the GPU
//! reads those CPU-written bytes as the presented frame, with no GPU draw.
//!
//! Run twice — pad neutral, then Cross held — the test asserts the running
//! guest produced different pixels (input → CPU 2D draw), that each flip
//! advanced the VideoOut flip count, and that the presented frame
//! (`session.last_image()`) reflects the drawn content and differs between the
//! two inputs. That whole chain — input, CPU 2D draw, flip, present — is the
//! M3 gate, and it is exercised by the guest actually executing, not by the
//! test calling HLE handlers directly.
//!
//! Windows-gated: the runtime's mechanism (`VirtualAlloc`/VEH) is Windows-only.

#![cfg(target_os = "windows")]

use raeen_firmware::{HLE_TRAMPOLINE_BASE, HleTrampoline, LinkedModule};
use raeen_hle::HleRegistry;
use raeen_kernel::{OrbisKernel, VideoOutBuffer, VideoOutBufferAttribute};
use raeen_runtime::{GUEST_ARENA_BASE, execute_linked};

// Guest image layout (offsets into `module.image`, identity-mapped at
// `GUEST_ARENA_BASE`).
const ENTRY_OFF: usize = 0x0;
const PAD_OFF: usize = 0x100; // ScePadData scratch (12 bytes used).
const DISPLAY_OFF: usize = 0x140; // 4x4 RGBA8 display buffer (64 bytes).
const SLOT_PAD: usize = 0x1C0; // trampoline pointer: scePadReadState.
const SLOT_FLIP: usize = 0x1C8; // trampoline pointer: sceVideoOutSubmitFlip.

const DISPLAY_W: u32 = 4;
const DISPLAY_H: u32 = 4;
/// `SCE_VIDEO_OUT_PIXEL_FORMAT_A8B8G8R8_SRGB` — memory byte order R,G,B,A.
const PIXEL_FORMAT_A8B8G8R8: u64 = 0x8000_2200;
/// `SCE_VIDEO_OUT_TILING_MODE_LINEAR`.
const TILING_LINEAR: u32 = 1;

const BLACK: u32 = 0xFF00_0000; // opaque black (buttons == 0)
const WHITE: u32 = 0xFFFF_FFFF; // opaque white (any button held)
const CROSS_BUTTON: u32 = 0x0000_4000;

/// Assemble the interactive-2D homebrew into a flat image.
fn build_homebrew() -> Vec<u8> {
    let mut image = vec![0u8; 0x300];
    let mut off = ENTRY_OFF;
    let mut call_sites: Vec<(usize, usize)> = Vec::new(); // (disp32_pos, slot_off)

    let pad_abs = (GUEST_ARENA_BASE + PAD_OFF as u64).to_le_bytes();
    let display_abs = (GUEST_ARENA_BASE + DISPLAY_OFF as u64).to_le_bytes();

    let emit = |image: &mut [u8], off: &mut usize, bytes: &[u8]| {
        image[*off..*off + bytes.len()].copy_from_slice(bytes);
        *off += bytes.len();
    };

    // scePadReadState(handle=1, data=&pad)
    emit(&mut image, &mut off, &[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1
    emit(&mut image, &mut off, &[0x48, 0xBE]); // mov rsi, imm64
    emit(&mut image, &mut off, &pad_abs);
    emit(&mut image, &mut off, &[0xFF, 0x15]); // call qword [rip+disp32]
    call_sites.push((off, SLOT_PAD));
    emit(&mut image, &mut off, &[0, 0, 0, 0]); // disp32 (patched below)

    // buttons = *(u32 *)pad
    emit(&mut image, &mut off, &[0x48, 0xB9]); // mov rcx, imm64
    emit(&mut image, &mut off, &pad_abs);
    emit(&mut image, &mut off, &[0x8B, 0x01]); // mov eax, [rcx]

    // edx = BLACK; if (buttons != 0) edx = WHITE
    emit(&mut image, &mut off, &[0xBA]); // mov edx, imm32
    emit(&mut image, &mut off, &BLACK.to_le_bytes());
    emit(&mut image, &mut off, &[0x85, 0xC0]); // test eax, eax
    emit(&mut image, &mut off, &[0x74, 0x05]); // je +5 (skip the WHITE mov)
    emit(&mut image, &mut off, &[0xBA]); // mov edx, imm32
    emit(&mut image, &mut off, &WHITE.to_le_bytes());

    // Fill 16 dwords (4x4) at the display buffer with edx.
    emit(&mut image, &mut off, &[0x48, 0xBF]); // mov rdi, imm64
    emit(&mut image, &mut off, &display_abs);
    emit(&mut image, &mut off, &[0xB9, 0x10, 0x00, 0x00, 0x00]); // mov ecx, 16
    // loop:
    emit(&mut image, &mut off, &[0x89, 0x17]); // mov [rdi], edx
    emit(&mut image, &mut off, &[0x48, 0x83, 0xC7, 0x04]); // add rdi, 4
    emit(&mut image, &mut off, &[0xFF, 0xC9]); // dec ecx
    emit(&mut image, &mut off, &[0x75, 0xF6]); // jnz loop (-10)

    // sceVideoOutSubmitFlip(handle=1, bufferIndex=0, flipMode=0, flipArg=0)
    emit(&mut image, &mut off, &[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1
    emit(&mut image, &mut off, &[0x31, 0xF6]); // xor esi, esi
    emit(&mut image, &mut off, &[0x31, 0xD2]); // xor edx, edx
    emit(&mut image, &mut off, &[0x31, 0xC9]); // xor ecx, ecx
    emit(&mut image, &mut off, &[0xFF, 0x15]); // call qword [rip+disp32]
    call_sites.push((off, SLOT_FLIP));
    emit(&mut image, &mut off, &[0, 0, 0, 0]); // disp32 (patched below)

    // return pixel 0 in RAX (interactivity proof read straight from the buffer)
    emit(&mut image, &mut off, &[0x48, 0xB9]); // mov rcx, imm64
    emit(&mut image, &mut off, &display_abs);
    emit(&mut image, &mut off, &[0x8B, 0x01]); // mov eax, [rcx]
    emit(&mut image, &mut off, &[0xC3]); // ret

    assert!(off <= PAD_OFF, "code must not overlap the pad buffer");

    // Patch each `call qword [rip+disp32]` to reach its 8-byte trampoline slot.
    for (disp_pos, slot_off) in call_sites {
        let rip_after = disp_pos as i64 + 4;
        let disp32 = (slot_off as i64 - rip_after) as i32;
        image[disp_pos..disp_pos + 4].copy_from_slice(&disp32.to_le_bytes());
    }

    // Trampoline slot 0 -> scePadReadState, slot 1 -> sceVideoOutSubmitFlip.
    image[SLOT_PAD..SLOT_PAD + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    image[SLOT_FLIP..SLOT_FLIP + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + 8).to_le_bytes());

    image
}

fn linked_homebrew() -> LinkedModule {
    LinkedModule {
        image: build_homebrew(),
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![
            HleTrampoline {
                library: "libScePad".into(),
                function: "scePadReadState".into(),
                addr: HLE_TRAMPOLINE_BASE,
            },
            HleTrampoline {
                library: "libSceVideoOut".into(),
                function: "sceVideoOutSubmitFlip".into(),
                addr: HLE_TRAMPOLINE_BASE + 8,
            },
        ],
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    }
}

/// Register the homebrew's display buffer as VideoOut slot 0 (handle 1) with a
/// linear RGBA8 attribute of the buffer's dimensions, so a flip presents it via
/// the present-from-guest-memory path.
fn register_display_buffer(kernel: &OrbisKernel) {
    kernel.video_out_buffers.insert(
        (1, 0),
        VideoOutBuffer {
            set_index: 0,
            address: GUEST_ARENA_BASE + DISPLAY_OFF as u64,
            metadata: 0,
            attribute: VideoOutBufferAttribute {
                pixel_format: PIXEL_FORMAT_A8B8G8R8,
                tiling_mode: TILING_LINEAR,
                width: DISPLAY_W,
                height: DISPLAY_H,
                option: 0,
                dcc_clear_color: 0,
                dcc_control: 0,
            },
        },
    );
}

/// One run of the homebrew with a given pad state. Returns
/// `(guest_rax, presented_frame, flip_count)`.
fn run_with_pad(
    kernel: &OrbisKernel,
    hle: &HleRegistry,
    buttons: u32,
) -> (u64, raeen_gpu::RenderedImage, u64) {
    let mut pad = [0u8; 12];
    pad[0..4].copy_from_slice(&buttons.to_le_bytes());
    pad[4] = 128; // sticks centered (well-formed ScePadData)
    pad[5] = 128;
    pad[6] = 128;
    pad[7] = 128;
    kernel.set_pad_state(pad);

    let linked = linked_homebrew();
    let rax = execute_linked(&linked, hle, kernel, ENTRY_OFF as u64, &[])
        .expect("interactive 2D homebrew runs to completion");

    // The session the run just installed is the process-global one; nothing
    // else in this single-file test binary installs another concurrently.
    let frame = raeen_gpu::AgcGpuSession::global()
        .last_image()
        .expect("the flip presented a frame");
    let flips = kernel
        .video_out_flip_count
        .load(std::sync::atomic::Ordering::Relaxed);
    (rax, frame, flips)
}

#[test]
fn interactive_2d_homebrew_input_draws_and_presents_a_frame() {
    let hle = HleRegistry::new(); // registers the real libScePad + libSceVideoOut
    let kernel = OrbisKernel::new();
    register_display_buffer(&kernel);

    // Input A: no buttons -> the guest fills the buffer black.
    let (rax_a, frame_a, flips_a) = run_with_pad(&kernel, &hle, 0);
    // Input B: Cross held -> the guest fills the buffer white.
    let (rax_b, frame_b, flips_b) = run_with_pad(&kernel, &hle, CROSS_BUTTON);

    // (1) The running guest wrote different pixels for the two inputs — read
    // straight out of the guest display buffer via the guest's own RAX.
    assert_eq!(rax_a, u64::from(BLACK), "neutral pad -> black fill");
    assert_eq!(rax_b, u64::from(WHITE), "Cross held -> white fill");
    assert_ne!(rax_a, rax_b, "input changed the CPU-drawn pixels");

    // (2) Each flip advanced the VideoOut flip count.
    assert_eq!(flips_a, 1, "first flip counted");
    assert_eq!(flips_b, 2, "second flip counted");

    // (3) Present-from-guest-memory made last_image() reflect the CPU-drawn
    // content, and it differs between the two inputs.
    assert_eq!(frame_a.width, DISPLAY_W);
    assert_eq!(frame_a.height, DISPLAY_H);
    assert_eq!(
        frame_a.pixel(0, 0),
        Some([0, 0, 0, 255]),
        "black frame presented for neutral input"
    );
    assert_eq!(
        frame_b.pixel(0, 0),
        Some([255, 255, 255, 255]),
        "white frame presented for Cross held"
    );
    assert_ne!(
        frame_a.pixels, frame_b.pixels,
        "the presented frame differs with input — interactive 2D end to end"
    );
}
