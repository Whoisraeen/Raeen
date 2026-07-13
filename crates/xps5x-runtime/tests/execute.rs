//! RT0 acceptance test (design doc §8): a fully synthetic homebrew module —
//! one `PT_LOAD` segment whose entry function is hand-assembled x86-64
//! (`call qword ptr [rip+disp32]; ret`) calling a single imported symbol —
//! flows through the **real** LM1 linker ([`link_module`]) so its import
//! relocation slot is patched with a genuine `HLE_TRAMPOLINE_BASE`-relative
//! trampoline address, then [`execute_linked`] runs it natively: the guest
//! `call` traps into the VEH, which dispatches to the HLE-registered
//! `sceTestSentinel` and resumes the guest, which returns whatever HLE
//! left in RAX.
//!
//! Entirely hand-built buffers; no real firmware bytes anywhere.
//!
//! Windows-gated: RT0's mechanism (`VirtualAlloc`/VEH) is Windows-only
//! (design doc §7/§9).

#![cfg(target_os = "windows")]

use xps5x_firmware::dynlib::nid::{NidDatabase, nid_of};
use xps5x_firmware::dynlib::{DynSymbol, DynlibData, SceRela};
use xps5x_firmware::{HLE_TRAMPOLINE_BASE, HleTrampoline, LinkedModule, ModuleRegistry, SprxModule, SprxSegment, link_module};
use xps5x_hle::{HleContext, HleRegistry};
use xps5x_kernel::OrbisKernel;
use xps5x_runtime::{RuntimeError, execute_linked};

const R_X86_64_JUMP_SLOT: u64 = 7;

/// Sentinel HLE function RT0's acceptance test dispatches to.
fn sentinel(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0xC0DE
}

/// Write `call qword ptr [rip+disp32]` (`FF 15 <disp32>`) followed by `ret`
/// (`C3`) into `buf` starting at `entry_off`, targeting the 8-byte pointer
/// slot at `slot_off` (both offsets within the same flat image).
fn write_entry_stub(buf: &mut [u8], entry_off: usize, slot_off: usize) {
    let rip_after_instr = entry_off as i64 + 6; // FF 15 <disp32> is 6 bytes.
    let disp32 = (slot_off as i64 - rip_after_instr) as i32;

    buf[entry_off] = 0xFF;
    buf[entry_off + 1] = 0x15;
    buf[entry_off + 2..entry_off + 6].copy_from_slice(&disp32.to_le_bytes());
    buf[entry_off + 6] = 0xC3; // ret
}

/// Build the `PT_LOAD` segment bytes (a stub entry function calling one
/// import through `slot_off`) plus the [`DynlibData`] declaring that import
/// (NID `import_nid`, symtab index 0, one `R_X86_64_JUMP_SLOT` relocation at
/// `slot_off`) and the [`SprxModule`] wrapping it.
fn build_synthetic_module(import_nid: u64, entry_off: usize, slot_off: usize) -> (SprxModule, DynlibData) {
    let mut image = vec![0u8; 0x100];
    write_entry_stub(&mut image, entry_off, slot_off);

    let module = SprxModule {
        name: "rt0TestModule".to_string(),
        e_type: 0xFE18, // ET_SCE_DYNAMIC
        segments: vec![SprxSegment {
            vaddr: 0,
            data: image,
            flags: 5, // R+X
            mem_size: 0x100,
        }],
        dynlib_data: None,
        relro: None,
        dynamic: None,
        entry: entry_off as u64,
    };

    let dynlib = DynlibData {
        symbols: vec![DynSymbol {
            nid: import_nid,
            value: 0,
            is_import: true,
        }],
        relocations: vec![SceRela {
            offset: slot_off as u64,
            info: R_X86_64_JUMP_SLOT, // r_sym = 0 (only symtab entry)
            addend: 0,
        }],
        ..Default::default()
    };

    (module, dynlib)
}

#[test]
fn sentinel_call_through_real_lm1_linker_dispatches_to_hle() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;

    let hle = HleRegistry::new();
    hle.register("libtest", "sceTestSentinel", sentinel);
    let import_nid = nid_of("sceTestSentinel");

    let (module, dynlib) = build_synthetic_module(import_nid, ENTRY_OFF, SLOT_OFF);

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);

    let linked = link_module(&module, &dynlib, &registry, &hle, 0x1_0000_0000)
        .expect("synthetic module links against the HLE-registered sentinel");
    assert_eq!(linked.hle_trampolines.len(), 1, "exactly one HLE import resolved");
    assert_eq!(linked.hle_trampolines[0].library, "libtest");
    assert_eq!(linked.hle_trampolines[0].function, "sceTestSentinel");
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).expect("native execution succeeds");
    assert_eq!(result, 0xC0DE, "guest RAX after the trapped HLE call is the sentinel's return value");
}

/// A guest `call` through a trampoline slot whose index has no
/// corresponding [`HleTrampoline`] entry (out of range of the module's
/// table) must surface as `Err(UnresolvedTrampoline(_))`, not hang or
/// silently return a stub value. Hand-constructs the [`LinkedModule`]
/// directly (rather than through `link_module`, which never itself produces
/// a dangling trampoline reference) to exercise the VEH's defensive
/// out-of-range handling.
#[test]
fn call_to_unmapped_trampoline_index_returns_unresolved() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;

    let hle = HleRegistry::new();

    let mut image = vec![0u8; 0x100];
    write_entry_stub(&mut image, ENTRY_OFF, SLOT_OFF);
    // The slot points at HLE_TRAMPOLINE_BASE (trampoline index 0), but
    // `hle_trampolines` below is empty -- index 0 has no mapping.
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: 0,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    assert_eq!(err, RuntimeError::UnresolvedTrampoline(HLE_TRAMPOLINE_BASE));
}

/// `execute_linked` rejects more than 6 arguments before doing any mapping
/// (design doc §5/§7) -- a trivial one-instruction (`ret`) module is enough
/// to prove the check happens up front.
#[test]
fn more_than_six_args_is_rejected() {
    let hle = HleRegistry::new();
    let linked = LinkedModule {
        image: vec![0xC3], // ret
        base: 0,
        unresolved: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: 0,
    };

    let kernel = OrbisKernel::new();
    let args = [1u64, 2, 3, 4, 5, 6, 7];
    let err = execute_linked(&linked, &hle, &kernel, 0, &args).unwrap_err();
    assert_eq!(err, RuntimeError::TooManyArgs);
}

/// RT1a acceptance test (design doc §7/§8's "genuine fault -> `Faulted`, not
/// a hang or silent pass"): a hand-mapped module (no `link_module` needed --
/// this entry never calls an HLE import) whose entry function is
/// `mov rax, [0]; ret` (`48 8B 04 25 00 00 00 00 C3`) -- a wild dereference
/// of address `0`, reliably unmapped, entirely outside the trampoline guard
/// region (`HLE_TRAMPOLINE_BASE`, a high fixed sentinel far from any
/// `VirtualAlloc`-chosen address). `execute_linked` must recover this as
/// `Err(Faulted { .. })` via the VEH's `RtlCaptureContext`-based restore
/// (`dispatch.rs`) instead of crashing the test process -- and the test
/// process proves it survived by going on to make an entirely ordinary RT0
/// call (through the ordinary sentinel-dispatch path) right after, in the
/// same test.
#[test]
fn genuine_wild_fault_recovers_as_faulted_then_process_keeps_running() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();

    let mut image = vec![0u8; 0x100];
    image[ENTRY_OFF..ENTRY_OFF + 9].copy_from_slice(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xC3]);

    let linked = LinkedModule {
        image,
        base: 0,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr } => {
            // `addr` is the *faulting instruction's* Rip (the mapped
            // entry's host address), not the wild pointer (`0`) it
            // dereferenced.
            assert_ne!(addr, 0, "Faulted::addr is the faulting Rip, which is a real mapped-image address");
        }
        other => panic!("expected Err(Faulted {{ .. }}), got {other:?}"),
    }

    // The process survived: prove it with a completely ordinary RT0 call,
    // in this same test/thread, right after the recovered fault. Also
    // proves `run`'s `CALL_LOCK`/`ACTIVE_CONTEXT`/VEH state was fully torn
    // down and re-armed correctly by the faulted call rather than left
    // corrupted.
    let sentinel_hle = HleRegistry::new();
    sentinel_hle.register("libtest", "sceTestSentinel", sentinel);
    let import_nid = nid_of("sceTestSentinel");
    let (module, dynlib) = build_synthetic_module(import_nid, 0x0, 0x10);
    let db = NidDatabase::from_hle_names(sentinel_hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &sentinel_hle, 0x1_0000_0000)
        .expect("synthetic module links against the HLE-registered sentinel");

    let sentinel_kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &sentinel_hle, &sentinel_kernel, 0x0, &[])
        .expect("native execution succeeds after a recovered fault");
    assert_eq!(result, 0xC0DE, "trampoline dispatch still works normally after a recovered fault");
}

/// Writes `mov edi, dst; mov esi, src; mov edx, n; call qword ptr
/// [rip+disp32]; mov eax, dword ptr [rip+disp32]; ret` into `buf` starting
/// at `entry_off`: sets up `memcpy(dst, src, n)`'s SysV integer argument
/// registers, calls through the import slot at `slot_off` (trapping into
/// the VEH, which dispatches to the real HLE `memcpy`), then — proof that
/// bytes actually moved, without needing any accessor beyond the guest's
/// own `RAX` return value — reads the first 4 bytes now sitting at `dst`
/// straight back out of the same mapped image and returns them.
fn write_memcpy_entry_stub(buf: &mut [u8], entry_off: usize, slot_off: usize, dst: u32, src: u32, n: u32) {
    let mut off = entry_off;

    buf[off] = 0xBF; // mov edi, imm32
    buf[off + 1..off + 5].copy_from_slice(&dst.to_le_bytes());
    off += 5;

    buf[off] = 0xBE; // mov esi, imm32
    buf[off + 1..off + 5].copy_from_slice(&src.to_le_bytes());
    off += 5;

    buf[off] = 0xBA; // mov edx, imm32
    buf[off + 1..off + 5].copy_from_slice(&n.to_le_bytes());
    off += 5;

    let call_off = off;
    let call_rip_after = call_off as i64 + 6;
    let call_disp32 = (slot_off as i64 - call_rip_after) as i32;
    buf[call_off] = 0xFF;
    buf[call_off + 1] = 0x15;
    buf[call_off + 2..call_off + 6].copy_from_slice(&call_disp32.to_le_bytes());
    off += 6;

    let load_off = off;
    let load_rip_after = load_off as i64 + 6;
    let load_disp32 = (dst as i64) - load_rip_after;
    buf[load_off] = 0x8B; // mov eax, [rip+disp32]
    buf[load_off + 1] = 0x05;
    buf[load_off + 2..load_off + 6].copy_from_slice(&(load_disp32 as i32).to_le_bytes());
    off += 6;

    buf[off] = 0xC3; // ret
}

/// The memcpy-through-the-runtime acceptance test (this milestone's proof
/// of real behavior, not just dispatch): a synthetic module's entry sets up
/// `memcpy(dst, src, 4)`'s argument registers and calls the real,
/// HLE-registered `libc::memcpy` (not a test-local sentinel) through the
/// genuine VEH trap-and-dispatch path. `memcpy`'s [`xps5x_hle::HleContext`]
/// gives it a [`xps5x_hle::GuestMemory`] view of this same mapped image
/// (design doc's dispatch-context milestone), so the call actually reads
/// `src`'s bytes and writes them to `dst` inside the mapped image — proven
/// by having the guest itself read `dst` back into `RAX` right after the
/// call and return it, rather than relying on any test-only accessor into
/// the (separately `VirtualAlloc`'d, not the original `Vec`) mapped image.
#[test]
fn memcpy_hle_call_moves_real_bytes_through_the_runtime() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x30;
    const SRC_OFF: u32 = 0x40;
    const DST_OFF: u32 = 0x80;
    const PAYLOAD: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];

    // `HleRegistry::new()` already registers `libc::memcpy` for real (no
    // test-local sentinel needed) — this test exercises the actual
    // production HLE implementation end to end.
    let hle = HleRegistry::new();
    let memcpy_nid = nid_of("memcpy");

    let mut image = vec![0u8; 0x100];
    write_memcpy_entry_stub(&mut image, ENTRY_OFF, SLOT_OFF, DST_OFF, SRC_OFF, PAYLOAD.len() as u32);
    image[SRC_OFF as usize..SRC_OFF as usize + PAYLOAD.len()].copy_from_slice(&PAYLOAD);
    // dst region (image[DST_OFF..]) is left zeroed, so a nonzero read-back
    // can only mean the HLE `memcpy` actually wrote it.

    let module = SprxModule {
        name: "rt-memcpy-test".to_string(),
        e_type: 0xFE18, // ET_SCE_DYNAMIC
        segments: vec![SprxSegment {
            vaddr: 0,
            data: image,
            flags: 7, // R+W+X: this segment is both the executed code and the memcpy'd data
            mem_size: 0x100,
        }],
        dynlib_data: None,
        relro: None,
        dynamic: None,
        entry: ENTRY_OFF as u64,
    };

    let dynlib = DynlibData {
        symbols: vec![DynSymbol {
            nid: memcpy_nid,
            value: 0,
            is_import: true,
        }],
        relocations: vec![SceRela {
            offset: SLOT_OFF as u64,
            info: R_X86_64_JUMP_SLOT, // r_sym = 0 (only symtab entry)
            addend: 0,
        }],
        ..Default::default()
    };

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, 0x3_0000_0000)
        .expect("memcpy import resolves against the built-in libc HLE registration");
    assert_eq!(linked.hle_trampolines.len(), 1, "exactly one HLE import resolved");
    assert_eq!(linked.hle_trampolines[0].library, "libc");
    assert_eq!(linked.hle_trampolines[0].function, "memcpy");
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).expect("native execution succeeds");

    let expected = u32::from_le_bytes(PAYLOAD) as u64;
    assert_eq!(
        result, expected,
        "guest RAX (dst read back after the real memcpy call) must equal the src payload — proving \
         HleContext + GuestMemory + VEH dispatch routed a real byte-for-byte copy, not a no-op stub"
    );
}
