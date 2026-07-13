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
use xps5x_hle::HleRegistry;
use xps5x_runtime::{RuntimeError, execute_linked};

const R_X86_64_JUMP_SLOT: u64 = 7;

/// Sentinel HLE function RT0's acceptance test dispatches to.
fn sentinel(_args: &[u64]) -> u64 {
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

    let result = execute_linked(&linked, &hle, ENTRY_OFF as u64, &[]).expect("native execution succeeds");
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
    };

    let err = execute_linked(&linked, &hle, ENTRY_OFF as u64, &[]).unwrap_err();
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
    };

    let args = [1u64, 2, 3, 4, 5, 6, 7];
    let err = execute_linked(&linked, &hle, 0, &args).unwrap_err();
    assert_eq!(err, RuntimeError::TooManyArgs);
}

// A genuine-wild-fault test (guest code writing through a deliberately
// invalid pointer, outside the trampoline guard region) is deferred: RT0's
// VEH intentionally passes such faults on with EXCEPTION_CONTINUE_SEARCH
// rather than converting them to `RuntimeError::Faulted` (see
// `dispatch.rs`'s module doc comment) -- there is currently no safe
// recovery point to resume Rust execution from after a genuine access
// violation, so a real one would terminate the test process rather than
// return an `Err`. Wiring up `Faulted` is left to a later milestone.
