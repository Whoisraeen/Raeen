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

use xps5x_firmware::dynlib::nid::{nid_of, NidDatabase};
use xps5x_firmware::dynlib::{DynSymbol, DynlibData, SceRela};
use xps5x_firmware::{
    link_module, HleTrampoline, LinkedModule, ModuleRegistry, SprxModule, SprxSegment, TlsTemplate,
    HLE_TRAMPOLINE_BASE,
};
use xps5x_hle::{HleContext, HleRegistry};
use xps5x_kernel::OrbisKernel;
use xps5x_runtime::{execute_linked, execute_process, RunOutcome, RuntimeError, GUEST_ARENA_BASE};

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
fn build_synthetic_module(
    import_nid: u64,
    entry_off: usize,
    slot_off: usize,
) -> (SprxModule, DynlibData) {
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
        tls: None,
        procparam: None,
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

    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("synthetic module links against the HLE-registered sentinel");
    assert_eq!(
        linked.hle_trampolines.len(),
        1,
        "exactly one HLE import resolved"
    );
    assert_eq!(linked.hle_trampolines[0].library, "libtest");
    assert_eq!(linked.hle_trampolines[0].function, "sceTestSentinel");
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");
    assert_eq!(
        result, 0xC0DE,
        "guest RAX after the trapped HLE call is the sentinel's return value"
    );
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
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
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
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: 0,
        tls: None,
        procparam_offset: None,
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
    image[ENTRY_OFF..ENTRY_OFF + 9]
        .copy_from_slice(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xC3]);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr } => {
            // `addr` is the *faulting instruction's* Rip (the mapped
            // entry's host address), not the wild pointer (`0`) it
            // dereferenced.
            assert_ne!(
                addr, 0,
                "Faulted::addr is the faulting Rip, which is a real mapped-image address"
            );
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
    let linked = link_module(&module, &dynlib, &registry, &sentinel_hle, GUEST_ARENA_BASE)
        .expect("synthetic module links against the HLE-registered sentinel");

    let sentinel_kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &sentinel_hle, &sentinel_kernel, 0x0, &[])
        .expect("native execution succeeds after a recovered fault");
    assert_eq!(
        result, 0xC0DE,
        "trampoline dispatch still works normally after a recovered fault"
    );
}

/// Writes `mov rdi, dst_addr; mov rsi, src_addr; mov edx, n; call qword ptr
/// [rip+disp32]; mov eax, dword ptr [rip+disp32]; ret` into `buf` starting
/// at `entry_off`: sets up `memcpy(dst_addr, src_addr, n)`'s SysV integer
/// argument registers with arena-absolute 64-bit guest addresses (the arena
/// is identity-mapped far above the 32-bit range, so `dst`/`src` need a
/// 64-bit immediate load, unlike RT0's original flat-image addressing),
/// calls through the import slot at `slot_off` (trapping into the VEH,
/// which dispatches to the real HLE `memcpy`), then — proof that bytes
/// actually moved, without needing any accessor beyond the guest's own
/// `RAX` return value — reads the first 4 bytes now sitting at `dst_off`
/// (the same location's plain in-image offset, for the RIP-relative load)
/// straight back out of the same mapped image and returns them.
fn write_memcpy_entry_stub(
    buf: &mut [u8],
    entry_off: usize,
    slot_off: usize,
    dst_off: usize,
    dst_addr: u64,
    src_addr: u64,
    n: u32,
) {
    let mut off = entry_off;

    buf[off] = 0x48; // REX.W
    buf[off + 1] = 0xBF; // mov rdi, imm64
    buf[off + 2..off + 10].copy_from_slice(&dst_addr.to_le_bytes());
    off += 10;

    buf[off] = 0x48; // REX.W
    buf[off + 1] = 0xBE; // mov rsi, imm64
    buf[off + 2..off + 10].copy_from_slice(&src_addr.to_le_bytes());
    off += 10;

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
    let load_disp32 = (dst_off as i64) - load_rip_after;
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
    write_memcpy_entry_stub(
        &mut image,
        ENTRY_OFF,
        SLOT_OFF,
        DST_OFF as usize,
        GUEST_ARENA_BASE + DST_OFF as u64,
        GUEST_ARENA_BASE + SRC_OFF as u64,
        PAYLOAD.len() as u32,
    );
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
        tls: None,
        procparam: None,
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
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("memcpy import resolves against the built-in libc HLE registration");
    assert_eq!(
        linked.hle_trampolines.len(),
        1,
        "exactly one HLE import resolved"
    );
    assert_eq!(linked.hle_trampolines[0].library, "libc");
    assert_eq!(linked.hle_trampolines[0].function, "memcpy");
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    let expected = u32::from_le_bytes(PAYLOAD) as u64;
    assert_eq!(
        result, expected,
        "guest RAX (dst read back after the real memcpy call) must equal the src payload — proving \
         HleContext + GuestMemory + VEH dispatch routed a real byte-for-byte copy, not a no-op stub"
    );
}

/// Writes `mov rdi, imm64; call qword ptr [rip+disp32]; ret` into `buf`
/// starting at `entry_off`: sets up `malloc(size)`'s SysV first integer
/// argument register, calls through the import slot at `slot_off` (trapping
/// into the VEH, which dispatches to the real HLE `malloc`), then returns
/// immediately — the guest's own `RAX` on return *is* `malloc`'s result,
/// with no further guest instructions needed.
fn write_malloc_entry_stub(buf: &mut [u8], entry_off: usize, slot_off: usize, size: u64) {
    let mut off = entry_off;

    buf[off] = 0x48; // REX.W
    buf[off + 1] = 0xBF; // mov rdi, imm64
    buf[off + 2..off + 10].copy_from_slice(&size.to_le_bytes());
    off += 10;

    let call_rip_after = off as i64 + 6;
    let call_disp32 = (slot_off as i64 - call_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call_disp32.to_le_bytes());
    off += 6;

    buf[off] = 0xC3; // ret
}

/// RT2a acceptance test, part 1 (design doc §6/§8): a synthetic module's
/// entry calls the real, HLE-registered `libc::malloc` through the genuine
/// VEH trap-and-dispatch path and simply returns whatever it got back in
/// `RAX` — `malloc`'s actual return value, unmediated by any further guest
/// code. Asserts the returned address falls inside the arena's heap region
/// (`[GUEST_ARENA_BASE + HEAP_OFFSET, GUEST_ARENA_BASE + STACK_OFFSET)`,
/// `arena.rs`'s layout constants), proving `malloc` allocated real guest
/// memory from the arena rather than returning a fixed sentinel.
#[test]
fn malloc_hle_call_returns_a_pointer_inside_the_heap_region() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x20;
    const MALLOC_SIZE: u64 = 0x40;

    // Mirrors `arena.rs`'s private `HEAP_OFFSET`/`STACK_OFFSET` constants —
    // duplicated here (this is a black-box integration test in a separate
    // crate-external file, with no access to `xps5x_runtime::arena`) so the
    // heap-region assertion below doesn't rely on any of that module's
    // internals being exported.
    const HEAP_OFFSET: u64 = 0x4000_0000;
    const STACK_OFFSET: u64 = 0x8000_0000;

    let hle = HleRegistry::new();
    let malloc_nid = nid_of("malloc");

    let mut image = vec![0u8; 0x100];
    write_malloc_entry_stub(&mut image, ENTRY_OFF, SLOT_OFF, MALLOC_SIZE);

    let module = SprxModule {
        name: "rt2a-malloc-test".to_string(),
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
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };

    let dynlib = DynlibData {
        symbols: vec![DynSymbol {
            nid: malloc_nid,
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
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("malloc import resolves against the built-in libc HLE registration");
    assert_eq!(
        linked.hle_trampolines.len(),
        1,
        "exactly one HLE import resolved"
    );
    assert_eq!(linked.hle_trampolines[0].library, "libc");
    assert_eq!(linked.hle_trampolines[0].function, "malloc");
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    assert_ne!(
        result, 0,
        "malloc must not return NULL for a small, easily satisfiable request"
    );
    let heap_start = GUEST_ARENA_BASE + HEAP_OFFSET;
    let heap_end = GUEST_ARENA_BASE + STACK_OFFSET;
    assert!(
        result >= heap_start && result < heap_end,
        "malloc'd address {result:#x} must fall inside the heap region [{heap_start:#x}, {heap_end:#x})"
    );
}

/// Writes a guest entry that: calls `malloc(0x40)` (arg in RDI) through
/// `slot_malloc_off`, stashes the returned pointer at `scratch_off` (a
/// RIP-relative store into the mapped image itself — this test has no other
/// scratch storage available to a hand-assembled stub), reloads it as
/// `memset`'s first argument, calls `memset(ptr, 0xAB, 0x40)` through
/// `slot_memset_off`, reloads the pointer once more, and reads byte 0 of the
/// block back with a zero-extending byte load into `EAX`, which becomes the
/// guest's return value.
fn write_malloc_memset_readback_stub(
    buf: &mut [u8],
    entry_off: usize,
    slot_malloc_off: usize,
    slot_memset_off: usize,
    scratch_off: usize,
    malloc_size: u64,
    memset_value: u32,
) {
    let mut off = entry_off;

    // mov rdi, malloc_size
    buf[off] = 0x48;
    buf[off + 1] = 0xBF;
    buf[off + 2..off + 10].copy_from_slice(&malloc_size.to_le_bytes());
    off += 10;

    // call qword ptr [rip+disp32]  ->  slot_malloc_off
    let call1_rip_after = off as i64 + 6;
    let call1_disp32 = (slot_malloc_off as i64 - call1_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call1_disp32.to_le_bytes());
    off += 6;

    // mov [rip+disp32], rax  -> scratch_off (stash the malloc'd pointer)
    let store_rip_after = off as i64 + 7;
    let store_disp32 = (scratch_off as i64 - store_rip_after) as i32;
    buf[off] = 0x48;
    buf[off + 1] = 0x89;
    buf[off + 2] = 0x05;
    buf[off + 3..off + 7].copy_from_slice(&store_disp32.to_le_bytes());
    off += 7;

    // mov rdi, [rip+disp32]  <- scratch_off (memset's dst arg)
    let load1_rip_after = off as i64 + 7;
    let load1_disp32 = (scratch_off as i64 - load1_rip_after) as i32;
    buf[off] = 0x48;
    buf[off + 1] = 0x8B;
    buf[off + 2] = 0x3D;
    buf[off + 3..off + 7].copy_from_slice(&load1_disp32.to_le_bytes());
    off += 7;

    // mov esi, memset_value
    buf[off] = 0xBE;
    buf[off + 1..off + 5].copy_from_slice(&memset_value.to_le_bytes());
    off += 5;

    // mov edx, malloc_size (low 32 bits; MALLOC_SIZE is small in this test)
    buf[off] = 0xBA;
    buf[off + 1..off + 5].copy_from_slice(&(malloc_size as u32).to_le_bytes());
    off += 5;

    // call qword ptr [rip+disp32]  -> slot_memset_off
    let call2_rip_after = off as i64 + 6;
    let call2_disp32 = (slot_memset_off as i64 - call2_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call2_disp32.to_le_bytes());
    off += 6;

    // mov rdi, [rip+disp32]  <- scratch_off (reload the pointer for read-back)
    let load2_rip_after = off as i64 + 7;
    let load2_disp32 = (scratch_off as i64 - load2_rip_after) as i32;
    buf[off] = 0x48;
    buf[off + 1] = 0x8B;
    buf[off + 2] = 0x3D;
    buf[off + 3..off + 7].copy_from_slice(&load2_disp32.to_le_bytes());
    off += 7;

    // movzx eax, byte [rdi]
    buf[off] = 0x0F;
    buf[off + 1] = 0xB6;
    buf[off + 2] = 0x07;
    off += 3;

    // ret
    buf[off] = 0xC3;
}

/// RT2a acceptance test, part 2 (design doc §6/§8's full payoff): a
/// synthetic module's entry calls the real, HLE-registered `libc::malloc`
/// to get a block of guest heap memory, calls the real `libc::memset` to
/// fill it with a known byte, and reads byte 0 of that same block straight
/// back out through the guest's own load instruction — proving malloc
/// returned real, dereferenceable guest memory, memset actually wrote
/// through it, and the write is visible back through the identity-mapped
/// arena, all via the genuine VEH trap-and-dispatch path (no test-only
/// accessor into the arena).
#[test]
fn malloc_then_memset_then_readback_proves_real_guest_heap_memory() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_MALLOC_OFF: usize = 0x80;
    const SLOT_MEMSET_OFF: usize = 0x88;
    const SCRATCH_OFF: usize = 0x90;
    const MALLOC_SIZE: u64 = 0x40;
    const MEMSET_VALUE: u32 = 0xAB;

    let hle = HleRegistry::new();
    let malloc_nid = nid_of("malloc");
    let memset_nid = nid_of("memset");

    let mut image = vec![0u8; 0x100];
    write_malloc_memset_readback_stub(
        &mut image,
        ENTRY_OFF,
        SLOT_MALLOC_OFF,
        SLOT_MEMSET_OFF,
        SCRATCH_OFF,
        MALLOC_SIZE,
        MEMSET_VALUE,
    );

    let module = SprxModule {
        name: "rt2a-malloc-memset-test".to_string(),
        e_type: 0xFE18, // ET_SCE_DYNAMIC
        segments: vec![SprxSegment {
            vaddr: 0,
            data: image,
            flags: 7, // R+W+X: this segment is both the executed code and the scratch slot
            mem_size: 0x100,
        }],
        dynlib_data: None,
        relro: None,
        dynamic: None,
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };

    let dynlib = DynlibData {
        symbols: vec![
            DynSymbol {
                nid: malloc_nid,
                value: 0,
                is_import: true,
            },
            DynSymbol {
                nid: memset_nid,
                value: 0,
                is_import: true,
            },
        ],
        relocations: vec![
            SceRela {
                offset: SLOT_MALLOC_OFF as u64,
                info: R_X86_64_JUMP_SLOT, // r_sym = 0 -> malloc
                addend: 0,
            },
            SceRela {
                offset: SLOT_MEMSET_OFF as u64,
                info: (1u64 << 32) | R_X86_64_JUMP_SLOT, // r_sym = 1 -> memset
                addend: 0,
            },
        ],
        ..Default::default()
    };

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("malloc/memset imports resolve against the built-in libc HLE registration");
    assert_eq!(
        linked.hle_trampolines.len(),
        2,
        "exactly two HLE imports resolved"
    );
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    assert_eq!(
        result, MEMSET_VALUE as u64,
        "guest RAX (byte 0 of the malloc'd block, read back after the real memset call) must equal the byte \
         memset wrote — proving the guest malloc'd real arena memory, memset actually wrote through it, and \
         the write is visible back through the identity-mapped GuestArena"
    );
}

/// Writes a guest entry that: calls `sceKernelMapFlexibleMemory(addrOut,
/// len, prot)` through `slot_map_off` — where `addrOut` is `scratch_addr`, an
/// *arena-absolute* guest address the HLE call can `ctx.mem.write` its
/// 8-byte result pointer to (this test uses the same in-image location, at
/// `scratch_off`, both as that absolute address and as the RIP-relative
/// reload target below, since the arena is identity-mapped) — reloads the
/// address `sceKernelMapFlexibleMemory` just wrote as `memset`'s first
/// argument, calls `memset(ptr, memset_value, len)` through
/// `slot_memset_off`, reloads the pointer once more, and reads byte 0 of the
/// block back with a zero-extending byte load into `EAX`, which becomes the
/// guest's return value. Mirrors [`write_malloc_memset_readback_stub`], with
/// `sceKernelMapFlexibleMemory`'s three-register call in place of `malloc`'s
/// one-register call and (crucially) no explicit store back to `scratch_off`
/// after the first call — unlike `malloc`, which returns its result in
/// `RAX`, `sceKernelMapFlexibleMemory` writes its result directly to
/// `addrOut` (i.e. `scratch_off`) itself.
#[allow(clippy::too_many_arguments)]
fn write_mmap_memset_readback_stub(
    buf: &mut [u8],
    entry_off: usize,
    slot_map_off: usize,
    slot_memset_off: usize,
    scratch_off: usize,
    scratch_addr: u64,
    len: u64,
    prot: u32,
    memset_value: u32,
) {
    let mut off = entry_off;

    // mov rdi, scratch_addr (sceKernelMapFlexibleMemory's addrOut arg)
    buf[off] = 0x48;
    buf[off + 1] = 0xBF;
    buf[off + 2..off + 10].copy_from_slice(&scratch_addr.to_le_bytes());
    off += 10;

    // mov esi, len (sceKernelMapFlexibleMemory's len arg)
    buf[off] = 0xBE;
    buf[off + 1..off + 5].copy_from_slice(&(len as u32).to_le_bytes());
    off += 5;

    // mov edx, prot (sceKernelMapFlexibleMemory's prot arg)
    buf[off] = 0xBA;
    buf[off + 1..off + 5].copy_from_slice(&prot.to_le_bytes());
    off += 5;

    // call qword ptr [rip+disp32]  -> slot_map_off
    let call1_rip_after = off as i64 + 6;
    let call1_disp32 = (slot_map_off as i64 - call1_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call1_disp32.to_le_bytes());
    off += 6;

    // mov rdi, [rip+disp32]  <- scratch_off (memset's dst arg: the address
    // sceKernelMapFlexibleMemory just wrote to addrOut)
    let load1_rip_after = off as i64 + 7;
    let load1_disp32 = (scratch_off as i64 - load1_rip_after) as i32;
    buf[off] = 0x48;
    buf[off + 1] = 0x8B;
    buf[off + 2] = 0x3D;
    buf[off + 3..off + 7].copy_from_slice(&load1_disp32.to_le_bytes());
    off += 7;

    // mov esi, memset_value
    buf[off] = 0xBE;
    buf[off + 1..off + 5].copy_from_slice(&memset_value.to_le_bytes());
    off += 5;

    // mov edx, len (memset's n arg)
    buf[off] = 0xBA;
    buf[off + 1..off + 5].copy_from_slice(&(len as u32).to_le_bytes());
    off += 5;

    // call qword ptr [rip+disp32]  -> slot_memset_off
    let call2_rip_after = off as i64 + 6;
    let call2_disp32 = (slot_memset_off as i64 - call2_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call2_disp32.to_le_bytes());
    off += 6;

    // mov rdi, [rip+disp32]  <- scratch_off (reload the pointer for read-back)
    let load2_rip_after = off as i64 + 7;
    let load2_disp32 = (scratch_off as i64 - load2_rip_after) as i32;
    buf[off] = 0x48;
    buf[off + 1] = 0x8B;
    buf[off + 2] = 0x3D;
    buf[off + 3..off + 7].copy_from_slice(&load2_disp32.to_le_bytes());
    off += 7;

    // movzx eax, byte [rdi]
    buf[off] = 0x0F;
    buf[off + 1] = 0xB6;
    buf[off + 2] = 0x07;
    off += 3;

    // ret
    buf[off] = 0xC3;
}

/// RT2b acceptance test (design doc §6/§8): a synthetic module's entry calls
/// the real, HLE-registered `libkernel::sceKernelMapFlexibleMemory` to map a
/// region from the arena's mmap region, calls the real `libc::memset` to
/// fill it with a known byte, and reads byte 0 of that same block straight
/// back out through the guest's own load instruction — proving
/// `sceKernelMapFlexibleMemory` now allocates real, dereferenceable arena
/// memory (RT2 Task 5; it no longer returns the old fake-address stub) and
/// that `memset` actually wrote through it, all via the genuine VEH
/// trap-and-dispatch path (no test-only accessor into the arena).
///
/// Separately — using the *same* [`OrbisKernel`] instance passed to
/// `execute_linked` — asserts that `kernel.memory.is_mapped` reflects the
/// mapping `sceKernelMapFlexibleMemory` recorded, and that the mapped
/// address falls inside the arena's mmap region (`arena.rs`'s private
/// `MMAP_OFFSET`/`ARENA_SPAN` constants, mirrored here for the same reason
/// as the RT2a heap-region test above: this is a black-box integration test
/// with no access to `xps5x_runtime::arena`'s internals).
#[test]
fn mmap_then_memset_then_readback_proves_real_arena_memory_and_records_vmm_metadata() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_MAP_OFF: usize = 0x80;
    const SLOT_MEMSET_OFF: usize = 0x88;
    const SCRATCH_OFF: usize = 0x90;
    const MMAP_LEN: u64 = 0x40;
    const MMAP_PROT: u32 = 0x3; // R+W
    const MEMSET_VALUE: u32 = 0xCD;

    const MMAP_OFFSET: u64 = 0xA000_0000;
    const ARENA_SPAN: u64 = 0x1_0000_0000;

    let hle = HleRegistry::new();
    let map_nid = nid_of("sceKernelMapFlexibleMemory");
    let memset_nid = nid_of("memset");

    let scratch_addr = GUEST_ARENA_BASE + SCRATCH_OFF as u64;

    let mut image = vec![0u8; 0x100];
    write_mmap_memset_readback_stub(
        &mut image,
        ENTRY_OFF,
        SLOT_MAP_OFF,
        SLOT_MEMSET_OFF,
        SCRATCH_OFF,
        scratch_addr,
        MMAP_LEN,
        MMAP_PROT,
        MEMSET_VALUE,
    );

    let module = SprxModule {
        name: "rt2b-mmap-memset-test".to_string(),
        e_type: 0xFE18, // ET_SCE_DYNAMIC
        segments: vec![SprxSegment {
            vaddr: 0,
            data: image,
            flags: 7, // R+W+X: this segment is both the executed code and the scratch slot
            mem_size: 0x100,
        }],
        dynlib_data: None,
        relro: None,
        dynamic: None,
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };

    let dynlib = DynlibData {
        symbols: vec![
            DynSymbol {
                nid: map_nid,
                value: 0,
                is_import: true,
            },
            DynSymbol {
                nid: memset_nid,
                value: 0,
                is_import: true,
            },
        ],
        relocations: vec![
            SceRela {
                offset: SLOT_MAP_OFF as u64,
                info: R_X86_64_JUMP_SLOT, // r_sym = 0 -> sceKernelMapFlexibleMemory
                addend: 0,
            },
            SceRela {
                offset: SLOT_MEMSET_OFF as u64,
                info: (1u64 << 32) | R_X86_64_JUMP_SLOT, // r_sym = 1 -> memset
                addend: 0,
            },
        ],
        ..Default::default()
    };

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE).expect(
        "sceKernelMapFlexibleMemory/memset imports resolve against the built-in HLE registration",
    );
    assert_eq!(
        linked.hle_trampolines.len(),
        2,
        "exactly two HLE imports resolved"
    );
    assert!(linked.unresolved.is_empty());

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    assert_eq!(
        result, MEMSET_VALUE as u64,
        "guest RAX (byte 0 of the mmap'd block, read back after the real memset call) must equal the byte \
         memset wrote — proving sceKernelMapFlexibleMemory allocated real, dereferenceable arena memory and \
         memset actually wrote through it"
    );

    // Separately, using the same `kernel` instance: the VMM must have
    // recorded the arena mapping's metadata (record_mapping tags it
    // "arena_mmap"), and the recorded address must fall inside the arena's
    // mmap region.
    let mapped_region = kernel
        .memory
        .dump_regions()
        .into_iter()
        .find(|region| region.name.as_deref() == Some("arena_mmap"))
        .expect("sceKernelMapFlexibleMemory must have recorded a VMM mapping via record_mapping");
    let mapped_addr = mapped_region.vaddr;

    assert!(
        kernel.memory.is_mapped(mapped_addr),
        "kernel.memory.is_mapped must reflect the arena mapping sceKernelMapFlexibleMemory recorded"
    );
    let mmap_start = GUEST_ARENA_BASE + MMAP_OFFSET;
    let mmap_end = GUEST_ARENA_BASE + ARENA_SPAN;
    assert!(
        mapped_addr >= mmap_start && mapped_addr < mmap_end,
        "mapped address {mapped_addr:#x} must fall inside the arena's mmap region [{mmap_start:#x}, {mmap_end:#x})"
    );
}

// --- RT2c-a: guest stack / RSP switch (design doc §5's acceptance tests) ---

/// Sentinel HLE function for the RSP-in-region test below: simply echoes
/// back its first argument (the guest RSP the entry stub captured into
/// `rdi` right before making this call), so the test can inspect it via the
/// guest's own returned `RAX` — no accessor into the runtime's internals is
/// needed.
fn capture_rsp(_ctx: &HleContext, args: &[u64]) -> u64 {
    args[0]
}

/// Writes `mov rdi, rsp` (`48 89 E7`) followed by `call qword ptr
/// [rip+disp32]` (`FF 15 <disp32>`) and `ret` (`C3`) into `buf` starting at
/// `entry_off`, targeting the 8-byte import slot at `slot_off`: the guest
/// captures its *own* current RSP into the HLE call's first SysV argument
/// register before trapping into the VEH, proving (via the trampoline's own
/// established arg-marshaling path) exactly what RSP the guest was running
/// on at the moment of the call.
fn write_rsp_capture_entry_stub(buf: &mut [u8], entry_off: usize, slot_off: usize) {
    let mut off = entry_off;

    // mov rdi, rsp
    buf[off] = 0x48;
    buf[off + 1] = 0x89;
    buf[off + 2] = 0xE7;
    off += 3;

    // call qword ptr [rip+disp32]  -> slot_off
    let call_rip_after = off as i64 + 6;
    let call_disp32 = (slot_off as i64 - call_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call_disp32.to_le_bytes());
    off += 6;

    // ret
    buf[off] = 0xC3;
}

/// RT2c-a acceptance test, part 1 (design doc §5): the guest entry captures
/// its own RSP into the HLE trampoline call's first argument register (`rdi`)
/// right before trapping into the VEH; the registered `sceCaptureRsp` HLE
/// function simply echoes that value back as its return, so it ends up in
/// the guest's own `RAX` and, after the stub's final `ret`, becomes
/// `execute_linked`'s result. Asserts that value falls inside the arena's
/// guest stack region — proving `entry` actually ran on the dedicated guest
/// stack (via `dispatch::run`'s `call_on_guest_stack` RSP switch), not on the
/// host thread's own stack.
///
/// Mirrors the RT2a heap-region test above: `STACK_OFFSET`/`STACK_SIZE` are
/// duplicated here (this is a black-box integration test with no access to
/// `xps5x_runtime::arena`'s private constants).
#[test]
fn guest_call_runs_on_dedicated_guest_stack_region() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;

    const STACK_OFFSET: u64 = 0x8000_0000;
    const STACK_SIZE: u64 = 0x2000_0000;

    let hle = HleRegistry::new();
    hle.register("libtest", "sceCaptureRsp", capture_rsp);
    let import_nid = nid_of("sceCaptureRsp");

    let mut image = vec![0u8; 0x100];
    write_rsp_capture_entry_stub(&mut image, ENTRY_OFF, SLOT_OFF);

    let module = SprxModule {
        name: "rt2c-rsp-region-test".to_string(),
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
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };

    let dynlib = DynlibData {
        symbols: vec![DynSymbol {
            nid: import_nid,
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
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("synthetic module links against the HLE-registered sceCaptureRsp");
    assert_eq!(
        linked.hle_trampolines.len(),
        1,
        "exactly one HLE import resolved"
    );

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    let stack_start = GUEST_ARENA_BASE + STACK_OFFSET;
    let stack_end = GUEST_ARENA_BASE + STACK_OFFSET + STACK_SIZE;
    assert!(
        result >= stack_start && result < stack_end,
        "guest RSP {result:#x} observed during the HLE call must fall inside the arena's guest stack region \
         [{stack_start:#x}, {stack_end:#x}) — proving entry ran on the dedicated guest stack, not the host stack"
    );
}

/// RT2c-a acceptance test, part 2 (design doc §5): a hand-assembled guest
/// entry that actually *uses* its stack — `sub rsp, 16; mov qword ptr [rsp],
/// 0x1234; mov rax, [rsp]; add rsp, 16; ret` — writing a value below RSP and
/// reading it back, proving the guest stack region is real, writable memory
/// and that `call_on_guest_stack`'s RSP switch/restore leaves the guest able
/// to push/pop and address its own locals correctly. No HLE import needed —
/// hand-mapped directly, like the RT1a fault-recovery test below.
#[test]
fn guest_stub_uses_real_guest_stack_memory_and_returns_correct_value() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;

    // sub rsp, 16
    image[off..off + 4].copy_from_slice(&[0x48, 0x83, 0xEC, 0x10]);
    off += 4;

    // mov qword ptr [rsp], 0x1234
    image[off..off + 8].copy_from_slice(&[0x48, 0xC7, 0x04, 0x24, 0x34, 0x12, 0x00, 0x00]);
    off += 8;

    // mov rax, [rsp]
    image[off..off + 4].copy_from_slice(&[0x48, 0x8B, 0x04, 0x24]);
    off += 4;

    // add rsp, 16
    image[off..off + 4].copy_from_slice(&[0x48, 0x83, 0xC4, 0x10]);
    off += 4;

    // ret
    image[off] = 0xC3;

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");
    assert_eq!(
        result, 0x1234,
        "a value written below RSP on the guest stack must read back correctly, proving the guest stack region \
         is real writable memory. (This stub returns 0x1234 whether it runs on the guest or host stack, so it \
         proves stack usability, not the switch itself — the switch is proven by \
         `guest_call_runs_on_dedicated_guest_stack_region` above; here a broken RSP *restore* would instead crash \
         the host after the guest's `ret`.)"
    );
}

/// RT2c-a robustness regression (design doc §7): a guest that returns
/// normally but leaves a callee-saved register (`r15`) clobbered must NOT
/// corrupt the host stack pointer. `call_on_guest_stack` saves/restores the
/// host RSP through a RIP-relative static slot, depending on *no*
/// general-purpose register surviving the guest `call` — so this runs to
/// completion and returns cleanly. Under the earlier design (which carried the
/// save-slot pointer in `r15` across the call), this stub's `mov r15, <wild>`
/// would have made the post-`call` `mov rsp, [r15]` restore load a
/// guest-controlled value into the host RSP, crashing the process. If this
/// test returns at all, the RIP-relative fix holds.
#[test]
fn guest_clobbering_r15_does_not_corrupt_host_rsp() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;

    // mov r15, 0xFFFFFFFFDEADBEEF  (49 C7 C7 imm32, sign-extended) — clobber a
    // callee-saved register with a wild value, without restoring it, then
    // return normally.
    image[off..off + 7].copy_from_slice(&[0x49, 0xC7, 0xC7, 0xEF, 0xBE, 0xAD, 0xDE]);
    off += 7;

    // mov eax, 0x1234  (B8 imm32) — the return value.
    image[off..off + 5].copy_from_slice(&[0xB8, 0x34, 0x12, 0x00, 0x00]);
    off += 5;

    // ret
    image[off] = 0xC3;

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");
    assert_eq!(
        result, 0x1234,
        "a guest that clobbers r15 and returns normally must not corrupt the host RSP — the RIP-relative \
         host-RSP restore must not depend on the guest preserving any register"
    );
}

// --- RT2c-b: TLS via fsbase (design doc §3/§5's acceptance tests) ---
//
// `xps5x_runtime::tls` is a private module, so this black-box test file
// duplicates the same minimal FSGSBASE probe/read it uses internally (the
// same pattern already used elsewhere in this file for private arena
// constants like `HEAP_OFFSET`/`STACK_OFFSET`) -- purely to (a) gate these
// tests on FSGSBASE actually being available on the machine running them,
// since that's environment-dependent, and (b) observe the *host* thread's FS
// base for the "no leak"/"restored" assertions, which requires reading it
// from outside `execute_linked` entirely.

/// Duplicate of `xps5x_runtime::tls::fsgsbase_available`'s CPUID probe.
fn fsgsbase_available() -> bool {
    // SAFETY: `__cpuid_count` is a safe fn on this target/rustc -- `CPUID`
    // is unconditionally available on x86-64; leaf 7/sub-leaf 0 is a
    // standard, always-queryable "Extended Features" leaf.
    let regs = core::arch::x86_64::__cpuid_count(7, 0);
    (regs.ebx & 1) != 0
}

/// Duplicate of `xps5x_runtime::tls::read_fsbase` -- reads the *host*
/// thread's current FS base, used only to observe that `execute_linked`
/// restores it after a guest call.
///
/// # Safety
/// Caller must have confirmed `fsgsbase_available()` returns `true` first.
unsafe fn read_host_fsbase() -> u64 {
    let value: u64;
    // SAFETY: per this function's contract, the caller has confirmed
    // FSGSBASE is available. Same `RDFSBASE RAX` encoding
    // (`F3 48 0F AE C0`) as `xps5x_runtime::tls::read_fsbase`.
    unsafe {
        core::arch::asm!(
            ".byte 0xf3, 0x48, 0x0f, 0xae, 0xc0", // rdfsbase rax
            out("rax") value,
            options(nomem, nostack, preserves_flags),
        );
    }
    value
}

/// A trivial hand-mapped module (`ret`) with no HLE imports -- reused by
/// several of the tests below that only care about `execute_linked`'s
/// TLS/fsbase side effects, not any particular guest computation.
fn trivial_ret_module() -> LinkedModule {
    LinkedModule {
        image: vec![0xC3], // ret
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: 0,
        tls: None,
        procparam_offset: None,
    }
}

/// RT2c-b acceptance test, part 1 (design doc §5): a hand-assembled guest
/// entry that is exactly `mov rax, fs:[0]; ret` (`64 48 8B 04 25 00 00 00 00
/// C3`, the RT2c-b task brief's stub) must return the TCB guest address
/// `GuestArena::setup_main_tcb` installed -- proving a real `fs:`-prefixed
/// guest load sees the FS base `dispatch::run` set via `WRFSBASE`.
///
/// `setup_main_tcb` is the very first heap allocation `execute_linked` makes
/// on a freshly constructed (hence freshly heap-bumped) `GuestArena`, so its
/// address is deterministically `GUEST_ARENA_BASE + HEAP_OFFSET` -- mirrors
/// the same private-constant-duplication pattern the RT2a heap-region test
/// above (`malloc_hle_call_returns_a_pointer_inside_the_heap_region`) uses,
/// for the same reason (black-box test, no access to `arena.rs`'s private
/// `HEAP_OFFSET`).
#[test]
fn guest_fs_zero_load_reads_the_installed_tcb() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping guest_fs_zero_load_reads_the_installed_tcb"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const HEAP_OFFSET: u64 = 0x4000_0000;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    image[ENTRY_OFF..ENTRY_OFF + 9]
        .copy_from_slice(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00]);
    image[ENTRY_OFF + 9] = 0xC3;

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    let expected_tcb = GUEST_ARENA_BASE + HEAP_OFFSET;
    println!(
        "guest_fs_zero_load_reads_the_installed_tcb: RAX = {result:#x}, expected TCB = {expected_tcb:#x}"
    );
    assert_eq!(
        result, expected_tcb,
        "guest `mov rax, fs:[0]` must read the TCB self-pointer `setup_main_tcb` installed"
    );
}

/// RT2c-b acceptance test, part 2 (design doc §5): a hand-assembled guest
/// entry that writes a known 64-bit value to `fs:[8]`, clears `rax`, then
/// reads it back through `fs:[8]` and returns it -- proving `fs:`-relative
/// addressing at a *non-zero* offset round-trips (not just the `fs:[0]`
/// self-pointer read above), i.e. real TLS-offset accesses (stack-protector
/// canary, TLS variables) resolve correctly through the FS base
/// `dispatch::run` set.
#[test]
fn guest_fs_offset_round_trip_writes_and_reads_back() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping guest_fs_offset_round_trip_writes_and_reads_back"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const TLS_VALUE: u64 = 0x1122_3344_5566_7788;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;

    // mov rax, TLS_VALUE (48 B8 imm64)
    image[off] = 0x48;
    image[off + 1] = 0xB8;
    image[off + 2..off + 10].copy_from_slice(&TLS_VALUE.to_le_bytes());
    off += 10;

    // mov fs:[8], rax (64 48 89 04 25 08 00 00 00)
    image[off..off + 9].copy_from_slice(&[0x64, 0x48, 0x89, 0x04, 0x25, 0x08, 0x00, 0x00, 0x00]);
    off += 9;

    // xor rax, rax (48 31 C0) -- prove the load below actually reloads a
    // fresh value rather than the assembler's own register state.
    image[off..off + 3].copy_from_slice(&[0x48, 0x31, 0xC0]);
    off += 3;

    // mov rax, fs:[8] (64 48 8B 04 25 08 00 00 00)
    image[off..off + 9].copy_from_slice(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x08, 0x00, 0x00, 0x00]);
    off += 9;

    // ret
    image[off] = 0xC3;

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("native execution succeeds");

    println!(
        "guest_fs_offset_round_trip_writes_and_reads_back: RAX = {result:#x}, expected = {TLS_VALUE:#x}"
    );
    assert_eq!(
        result, TLS_VALUE,
        "a value written to fs:[8] must read back through fs:[8], proving TLS-offset addressing (not just \
         fs:[0]) round-trips correctly"
    );
}

/// RT2c-b acceptance test, part 3 (design doc §7's restore requirement): the
/// *host* thread's FS base must be back to its pre-call value after
/// `execute_linked` returns -- proving `dispatch::run`'s shared continuation
/// restore (`write_fsbase(ctx.orig_fsbase.get())`) actually runs. Calls
/// `execute_linked` twice in a row to also prove no leak accumulates across
/// repeated calls (each call sets fsbase to a *different* TCB address, since
/// each builds a fresh `GuestArena`, so a broken restore that merely "worked
/// once" would still be caught by the second call).
#[test]
fn host_fsbase_is_restored_after_execute_linked_returns() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping host_fsbase_is_restored_after_execute_linked_returns"
        );
        return;
    }

    // SAFETY: `fsgsbase_available()` just returned `true` above.
    let before = unsafe { read_host_fsbase() };

    let hle = HleRegistry::new();
    let kernel = OrbisKernel::new();
    let _ = execute_linked(&trivial_ret_module(), &hle, &kernel, 0, &[])
        .expect("native execution succeeds");

    // SAFETY: same as above.
    let after_first = unsafe { read_host_fsbase() };
    println!(
        "host_fsbase_is_restored_after_execute_linked_returns: before={before:#x} after_first={after_first:#x}"
    );
    assert_eq!(
        after_first, before,
        "host FS base must be restored to its pre-call value after execute_linked returns"
    );

    let hle2 = HleRegistry::new();
    let kernel2 = OrbisKernel::new();
    let _ = execute_linked(&trivial_ret_module(), &hle2, &kernel2, 0, &[])
        .expect("native execution succeeds");

    // SAFETY: same as above.
    let after_second = unsafe { read_host_fsbase() };
    println!(
        "host_fsbase_is_restored_after_execute_linked_returns: after_second={after_second:#x}"
    );
    assert_eq!(
        after_second, before,
        "host FS base must still equal the pre-call value after a second, independent execute_linked call"
    );
}

/// RT2c-b + RT1a interaction test (design doc §5's "fault path still
/// restores fsbase"): reuses the genuine-wild-fault shape
/// (`genuine_wild_fault_recovers_as_faulted_then_process_keeps_running`
/// above) but additionally asserts the *host* FS base is restored to its
/// pre-call value even when the guest call is recovered via RT1a's
/// `RtlCaptureContext`-based fault path, not just on the ordinary-return
/// path covered by the test above -- proving the restore in `dispatch::run`'s
/// shared continuation runs on *both* arrivals, exactly as documented.
#[test]
fn host_fsbase_is_restored_after_a_recovered_genuine_fault() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping host_fsbase_is_restored_after_a_recovered_genuine_fault"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;

    // SAFETY: `fsgsbase_available()` just returned `true` above.
    let before = unsafe { read_host_fsbase() };

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    image[ENTRY_OFF..ENTRY_OFF + 9]
        .copy_from_slice(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xC3]);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { .. } => {}
        other => panic!("expected Err(Faulted {{ .. }}), got {other:?}"),
    }

    // SAFETY: same as above.
    let after = unsafe { read_host_fsbase() };
    println!(
        "host_fsbase_is_restored_after_a_recovered_genuine_fault: before={before:#x} after={after:#x}"
    );
    assert_eq!(
        after, before,
        "host FS base must be restored even after a recovered genuine guest fault (RT1a path), not just on \
         ordinary return"
    );
}

// --- Wall #1 (crt0 / process environment): W1a process stack + `_start`
// entry, W1b `exit()` termination (design doc
// `2026-07-13-xps5x-crt0-process-env-design.md` §6/§7) ---
//
// `execute_process` enters `_start` via a `jmp`, not a `call` (`stack.rs`'s
// `enter_guest_at_start`), so the guest's first instruction sees `rsp`
// pointing directly at `argc` -- no pushed return address for a plain `ret`
// to safely pop. A hand-assembled `_start` therefore can't "return" a value
// the way a function-mode stub (`execute_linked`'s tests, above) does: a
// `ret` here would pop `argc` itself and jump to it as if it were code. This
// is exactly the design doc's documented "malformed `_start`" case -- caught
// by the existing RT1a genuine-fault recovery, not a new mechanism -- and it
// doubles as this milestone's observation channel: jumping deliberately to
// a value read off the process stack makes that value observable as
// `RuntimeError::Faulted { addr }`, without inventing any new return path.

/// Writes `mov rax, [rsp]` (`48 8B 04 24`) followed by `jmp rax` (`FF E0`)
/// into `buf` at `entry_off`: reads `argc` off the process stack and jumps
/// to it as an address, faulting (Rip == argc) -- the W1a "prove argc is
/// ABI-correct" stub.
fn write_start_read_argc_and_jump_stub(buf: &mut [u8], entry_off: usize) {
    buf[entry_off..entry_off + 4].copy_from_slice(&[0x48, 0x8B, 0x04, 0x24]); // mov rax, [rsp]
    buf[entry_off + 4..entry_off + 6].copy_from_slice(&[0xFF, 0xE0]); // jmp rax
}

/// Writes `mov rax, [rsp+8]` (`48 8B 44 24 08`), `movzx eax, byte [rax]`
/// (`0F B6 00`), then `jmp rax` (`FF E0`) into `buf` at `entry_off`: reads
/// `argv[0]`'s pointer off the process stack, loads that string's first
/// byte, and jumps to it as an address, faulting (Rip == that byte) -- the
/// W1a "prove the argv pointer table + strings are correct" stub.
fn write_start_read_argv0_byte_and_jump_stub(buf: &mut [u8], entry_off: usize) {
    buf[entry_off..entry_off + 5].copy_from_slice(&[0x48, 0x8B, 0x44, 0x24, 0x08]); // mov rax, [rsp+8]
    buf[entry_off + 5..entry_off + 8].copy_from_slice(&[0x0F, 0xB6, 0x00]); // movzx eax, byte [rax]
    buf[entry_off + 8..entry_off + 10].copy_from_slice(&[0xFF, 0xE0]); // jmp rax
}

/// W1a acceptance test, part 1 (design doc §6/§8): a hand-assembled `_start`
/// stub reads `argc` from `[rsp]` and jumps to it as an address. With a
/// single `argv` entry (`argc == 1`), that fault's `Rip` is observably `1` --
/// proving `build_process_stack` + `enter_guest_at_start` deliver a real,
/// ABI-correct `argc` at the guest's very first instruction.
#[test]
fn start_stub_observes_argc_equal_to_one_via_the_process_stack() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    write_start_read_argc_and_jump_stub(&mut image, ENTRY_OFF);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr } => {
            assert_eq!(
                addr, 1,
                "observed argc must equal 1 (a single argv entry was passed)"
            );
        }
        other => panic!("expected Err(Faulted {{ .. }}), got {other:?}"),
    }
}

/// W1a acceptance test, part 2 (design doc §6/§8): a hand-assembled `_start`
/// stub reads `argv[0]` from `[rsp+8]`, loads that string's first byte, and
/// jumps to it as an address. With `argv[0] == "/app/eboot.bin"`, that
/// fault's `Rip` is observably `0x2F` (`'/'`) -- proving the argv pointer
/// table and the strings it points to are both laid out correctly.
#[test]
fn start_stub_observes_argv0_first_byte_via_the_process_stack() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    write_start_read_argv0_byte_and_jump_stub(&mut image, ENTRY_OFF);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr } => {
            assert_eq!(addr, 0x2F, "observed argv[0][0] must equal '/' (0x2F)");
        }
        other => panic!("expected Err(Faulted {{ .. }}), got {other:?}"),
    }
}

/// Writes `mov edi, code` (`BF <imm32>`) followed by `call qword ptr
/// [rip+disp32]` (`FF 15 <disp32>`) into `buf` at `entry_off`, targeting the
/// 8-byte trampoline slot at `slot_off`: sets up `exit(code)`'s SysV first
/// integer argument register and calls through the trampoline slot. Nothing
/// follows the `call` -- a terminating call never resumes the guest (design
/// doc §4), so there is no "after" for this stub to reach.
fn write_start_exit_stub(buf: &mut [u8], entry_off: usize, slot_off: usize, code: u32) {
    let mut off = entry_off;

    buf[off] = 0xBF; // mov edi, imm32
    buf[off + 1..off + 5].copy_from_slice(&code.to_le_bytes());
    off += 5;

    let call_rip_after = off as i64 + 6;
    let call_disp32 = (slot_off as i64 - call_rip_after) as i32;
    buf[off] = 0xFF;
    buf[off + 1] = 0x15;
    buf[off + 2..off + 6].copy_from_slice(&call_disp32.to_le_bytes());
}

/// W1b acceptance test (design doc §6/§8): a hand-assembled `_start` stub
/// calls `exit(0x2A)` through its HLE trampoline slot. `veh_callback`
/// recognizes `libc::exit` as a terminating function (design doc §4) and
/// performs the exit-longjmp instead of servicing-and-resuming, so
/// `execute_process` returns `Ok(RunOutcome::Exited(0x2A))` -- not
/// `Returned`, and not a hang. The trampoline entry is hand-built directly
/// (as `call_to_unmapped_trampoline_index_returns_unresolved` above does),
/// bypassing `link_module`, since this test only needs the VEH to resolve
/// `(library, function) == ("libc", "exit")` at the guarded slot, not a full
/// NID-based link. The process surviving is proven by making an entirely
/// ordinary `execute_linked` call right after, in the same test.
#[test]
fn start_stub_calling_exit_returns_exited_with_the_given_code() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;
    const EXIT_CODE: u32 = 0x2A;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    write_start_exit_stub(&mut image, ENTRY_OFF, SLOT_OFF, EXIT_CODE);
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("exit() must not fault");
    assert_eq!(
        outcome,
        RunOutcome::Exited(EXIT_CODE as u64),
        "execute_process must report the exit code passed to exit(), not a normal return"
    );

    // The process survived: an entirely ordinary execute_linked call, in
    // this same test/thread, right after the exit-longjmp -- also proves
    // `run`'s CALL_LOCK/ACTIVE_CONTEXT/VEH state was fully torn down and
    // re-armed correctly, exactly like the RT1a fault-recovery test above.
    let sentinel_hle = HleRegistry::new();
    sentinel_hle.register("libtest", "sceTestSentinel", sentinel);
    let import_nid = nid_of("sceTestSentinel");
    let (module, dynlib) = build_synthetic_module(import_nid, 0x0, 0x10);
    let db = NidDatabase::from_hle_names(sentinel_hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked2 = link_module(&module, &dynlib, &registry, &sentinel_hle, GUEST_ARENA_BASE)
        .expect("synthetic module links against the HLE-registered sentinel");

    let sentinel_kernel = OrbisKernel::new();
    let result = execute_linked(&linked2, &sentinel_hle, &sentinel_kernel, 0x0, &[])
        .expect("native execution succeeds after an exit-longjmp");
    assert_eq!(
        result, 0xC0DE,
        "trampoline dispatch still works normally after an exit-longjmp"
    );
}

/// W1b regression (design doc §7): a `_start` that genuinely faults (not
/// via the argc/argv-jump trick above, but a real wild dereference, mirroring
/// the RT1a `execute_linked` test) must still return `Err(Faulted { .. })`
/// through `execute_process`, exactly as it does through `execute_linked`.
#[test]
fn start_stub_wild_fault_still_recovers_as_faulted_through_execute_process() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    // mov rax, [0]  -- a wild dereference of address 0, reliably unmapped.
    image[ENTRY_OFF..ENTRY_OFF + 9]
        .copy_from_slice(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xC3]);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr } => {
            assert_ne!(
                addr, 0,
                "Faulted::addr is the faulting Rip, a real mapped-image address"
            );
        }
        other => panic!("expected Err(Faulted {{ .. }}), got {other:?}"),
    }
}

/// W1b + RT2c-b interaction (design doc §7): the *host* FS base must be
/// restored to its pre-call value after `execute_process` returns, exactly
/// as `execute_linked` already guarantees (mirrors
/// `host_fsbase_is_restored_after_execute_linked_returns` above) -- proving
/// the exit-longjmp still reaches `run`'s shared fsbase-restore
/// continuation (design doc §4's "keep the fsbase-restore/host-RSP
/// discipline" requirement) rather than skipping it.
#[test]
fn execute_process_restores_host_fsbase_after_an_exit_longjmp() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping execute_process_restores_host_fsbase_after_an_exit_longjmp"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;
    const EXIT_CODE: u32 = 7;

    // SAFETY: `fsgsbase_available()` just returned `true` above.
    let before = unsafe { read_host_fsbase() };

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    write_start_exit_stub(&mut image, ENTRY_OFF, SLOT_OFF, EXIT_CODE);
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("exit() must not fault");
    assert_eq!(outcome, RunOutcome::Exited(EXIT_CODE as u64));

    // SAFETY: same as above.
    let after = unsafe { read_host_fsbase() };
    println!(
        "execute_process_restores_host_fsbase_after_an_exit_longjmp: before={before:#x} after={after:#x}"
    );
    assert_eq!(
        after, before,
        "host FS base must be restored after execute_process returns, even via the exit-longjmp path"
    );
}

// --- M1-B (wall #2): PT_TLS materialization + TPOFF64 + fs:0x28 canary ----
//
// These are the acceptance tests for the TLS wall: a module with a real
// `PT_TLS` template and a real `TPOFF64` relocation — linked through the
// genuine LM1 linker, not hand-patched — reads its `.tdata` value back
// through an fs-relative access at the linker-computed offset; and a
// stack-protected-style read of `fs:0x28` observes a real, nonzero canary
// with glibc's zero terminator byte.

/// A module whose `_start` loads the linker-resolved `TPOFF64` offset from a
/// data slot, reads the TLS variable through `fs:[rax]`, and exits with it:
///
/// ```text
/// mov rax, [rip -> slot_tls]   ; 48 8B 05 <disp32>   the tpoff (negative)
/// mov rdi, fs:[rax]            ; 64 48 8B 38         the TLS variable
/// call [rip -> slot_exit]      ; FF 15 <disp32>      exit(rdi)
/// ```
#[test]
fn tls_variable_read_through_linker_computed_tpoff64_round_trips_tdata() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping tls_variable_read_through_linker_computed_tpoff64_round_trips_tdata"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const SLOT_EXIT_OFF: usize = 0x40;
    const SLOT_TLS_OFF: usize = 0x48;
    const TLS_VALUE: u64 = 0x5AFE_C0DE;
    const TLS_VAR_OFF: u64 = 0x8; // the variable's offset inside the template
    const R_X86_64_TPOFF64: u64 = 18;

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;
    // mov rax, [rip+disp32] -> SLOT_TLS_OFF
    let disp_tls = (SLOT_TLS_OFF as i64 - (off as i64 + 7)) as i32;
    image[off..off + 3].copy_from_slice(&[0x48, 0x8B, 0x05]);
    image[off + 3..off + 7].copy_from_slice(&disp_tls.to_le_bytes());
    off += 7;
    // mov rdi, fs:[rax]
    image[off..off + 4].copy_from_slice(&[0x64, 0x48, 0x8B, 0x38]);
    off += 4;
    // call qword ptr [rip+disp32] -> SLOT_EXIT_OFF (never returns)
    let disp_exit = (SLOT_EXIT_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp_exit.to_le_bytes());

    // `.tdata`: 16 file-backed bytes with TLS_VALUE at TLS_VAR_OFF;
    // `mem_size` 0x18 adds 8 bytes of `.tbss`. block_size() =
    // align_up(0x18, max(0x10, 16)) = 0x20, so the linker must write
    // TLS_VAR_OFF - 0x20 = -0x18 into the TPOFF64 slot.
    let mut tdata = vec![0u8; 0x10];
    tdata[TLS_VAR_OFF as usize..TLS_VAR_OFF as usize + 8].copy_from_slice(&TLS_VALUE.to_le_bytes());

    let module = SprxModule {
        name: "tlsTestModule".to_string(),
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
        entry: ENTRY_OFF as u64,
        tls: Some(TlsTemplate {
            vaddr: 0,
            data: tdata,
            mem_size: 0x18,
            align: 0x10,
        }),
        procparam: None,
    };

    let dynlib = DynlibData {
        symbols: vec![DynSymbol {
            nid: nid_of("exit"),
            value: 0,
            is_import: true,
        }],
        relocations: vec![
            SceRela {
                offset: SLOT_EXIT_OFF as u64,
                info: R_X86_64_JUMP_SLOT, // r_sym = 0 -> the exit import
                addend: 0,
            },
            SceRela {
                offset: SLOT_TLS_OFF as u64,
                info: R_X86_64_TPOFF64, // r_sym = 0: offset entirely in the addend
                addend: TLS_VAR_OFF as i64,
            },
        ],
        ..Default::default()
    };

    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("a module with a TPOFF64 relocation must link without hard-failing (M1-B)");

    // The linker's half of the contract, asserted directly: the TPOFF64
    // slot holds the negative, block-relative fs offset.
    let slot = u64::from_le_bytes(
        linked.image[SLOT_TLS_OFF..SLOT_TLS_OFF + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        slot,
        (-0x18i64) as u64,
        "TPOFF64 slot must hold TLS_VAR_OFF - block_size()"
    );

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("TLS read must not fault");
    assert_eq!(
        outcome,
        RunOutcome::Exited(TLS_VALUE),
        "the guest must read its .tdata value back through the linker-computed fs-relative offset"
    );
}

/// M1-B canary acceptance: a `_start` that reads `fs:0x28` — exactly what a
/// stack-protector prologue does — and exits with it must observe a real,
/// nonzero canary whose low byte is zero (glibc's terminator convention).
/// A zeroed TCB would make stack-protected code "work" by coincidence; this
/// pins the honest ABI contract instead (the m1-homebrew anti-pattern).
#[test]
fn stack_chk_guard_canary_at_fs_0x28_is_nonzero_with_terminator_byte() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping stack_chk_guard_canary_at_fs_0x28_is_nonzero_with_terminator_byte"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x20;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    // mov rdi, fs:[0x28]  (64 48 8B 3C 25 28 00 00 00)
    image[ENTRY_OFF..ENTRY_OFF + 9]
        .copy_from_slice(&[0x64, 0x48, 0x8B, 0x3C, 0x25, 0x28, 0x00, 0x00, 0x00]);
    // call qword ptr [rip+disp32] -> SLOT_OFF
    let off = ENTRY_OFF + 9;
    let disp = (SLOT_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        unresolved: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam_offset: None,
    };

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("canary read must not fault");
    let RunOutcome::Exited(canary) = outcome else {
        panic!("expected the guest to exit with the canary value, got {outcome:?}");
    };
    assert_ne!(
        canary, 0,
        "fs:0x28 must hold a real, nonzero __stack_chk_guard canary (no zero-canary soft-success)"
    );
    assert_eq!(
        canary & 0xFF,
        0,
        "the canary's low byte is the glibc-style NUL terminator"
    );
}

// --- M1-C (wall #3): printf with real guest strings -> observable output --

/// The M1-C acceptance test at the runtime layer: a `_start`-shaped module
/// calls `printf("hello %s, %d!\n", "world", 42)` — the format string and
/// the `%s` pointee are real bytes in the module's own image, addressed
/// RIP-relative — then exits. The formatted output must land, byte-exact,
/// in the kernel's captured console: guest memory -> HLE printf -> host-
/// observable stdout, end to end through the genuine LM1 link + VEH
/// dispatch path.
#[test]
fn printf_with_guest_format_string_lands_in_the_kernel_console() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_PRINTF_OFF: usize = 0x80;
    const SLOT_EXIT_OFF: usize = 0x88;
    const FMT_OFF: usize = 0x90;
    const WORLD_OFF: usize = 0xB0;

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;

    // lea rdi, [rip+disp32] -> FMT_OFF
    let disp = (FMT_OFF as i64 - (off as i64 + 7)) as i32;
    image[off..off + 3].copy_from_slice(&[0x48, 0x8D, 0x3D]);
    image[off + 3..off + 7].copy_from_slice(&disp.to_le_bytes());
    off += 7;
    // lea rsi, [rip+disp32] -> WORLD_OFF
    let disp = (WORLD_OFF as i64 - (off as i64 + 7)) as i32;
    image[off..off + 3].copy_from_slice(&[0x48, 0x8D, 0x35]);
    image[off + 3..off + 7].copy_from_slice(&disp.to_le_bytes());
    off += 7;
    // mov edx, 42
    image[off] = 0xBA;
    image[off + 1..off + 5].copy_from_slice(&42u32.to_le_bytes());
    off += 5;
    // call qword ptr [rip+disp32] -> SLOT_PRINTF_OFF
    let disp = (SLOT_PRINTF_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());
    off += 6;
    // mov edi, 0 ; call [rip -> SLOT_EXIT_OFF]  (exit(0), never returns)
    image[off] = 0xBF;
    image[off + 1..off + 5].copy_from_slice(&0u32.to_le_bytes());
    off += 5;
    let disp = (SLOT_EXIT_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());

    image[FMT_OFF..FMT_OFF + 20].copy_from_slice(b"hello %s, %d!\n\0\0\0\0\0\0");
    image[WORLD_OFF..WORLD_OFF + 6].copy_from_slice(b"world\0");

    let module = SprxModule {
        name: "printfTestModule".to_string(),
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
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };
    let dynlib = DynlibData {
        symbols: vec![
            DynSymbol {
                nid: nid_of("printf"),
                value: 0,
                is_import: true,
            },
            DynSymbol {
                nid: nid_of("exit"),
                value: 0,
                is_import: true,
            },
        ],
        relocations: vec![
            SceRela {
                offset: SLOT_PRINTF_OFF as u64,
                info: R_X86_64_JUMP_SLOT, // r_sym = 0 -> printf
                addend: 0,
            },
            SceRela {
                offset: SLOT_EXIT_OFF as u64,
                info: (1u64 << 32) | R_X86_64_JUMP_SLOT, // r_sym = 1 -> exit
                addend: 0,
            },
        ],
        ..Default::default()
    };

    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("printf/exit imports must link");
    assert!(
        linked.unresolved.is_empty(),
        "printf and exit must both resolve to HLE"
    );

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("printf module must not fault");
    assert_eq!(outcome, RunOutcome::Exited(0));
    assert_eq!(
        kernel.console.contents(),
        "hello world, 42!\n",
        "the guest's printf output must be captured byte-exact in the kernel console"
    );
}

// --- M1-D (wall #4): sceKernelLoadStartModule pseudo-handle path ---------

/// M1-D acceptance at the runtime layer: a `_start`-shaped module calls
/// `sceKernelLoadStartModule("/system/common/lib/libSceSysmodule.sprx",
/// 0, 0, 0, 0, NULL)` — the path string lives in the module's own image —
/// and exits with the returned handle. On a fresh kernel the first
/// registered module gets handle 1, so `Exited(1)` proves the guest path
/// string was read, the module table was consulted, and a valid pseudo-
/// handle came back through the genuine link + VEH dispatch path.
#[test]
fn load_start_module_from_guest_returns_a_usable_handle() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_LSM_OFF: usize = 0x80;
    const SLOT_EXIT_OFF: usize = 0x88;
    const PATH_OFF: usize = 0x90;

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;

    // lea rdi, [rip+disp32] -> PATH_OFF
    let disp = (PATH_OFF as i64 - (off as i64 + 7)) as i32;
    image[off..off + 3].copy_from_slice(&[0x48, 0x8D, 0x3D]);
    image[off + 3..off + 7].copy_from_slice(&disp.to_le_bytes());
    off += 7;
    // xor esi,esi ; xor edx,edx ; xor ecx,ecx ; xor r8d,r8d ; xor r9d,r9d
    image[off..off + 12].copy_from_slice(&[
        0x31, 0xF6, 0x31, 0xD2, 0x31, 0xC9, 0x45, 0x31, 0xC0, 0x45, 0x31, 0xC9,
    ]);
    off += 12;
    // call qword ptr [rip+disp32] -> SLOT_LSM_OFF
    let disp = (SLOT_LSM_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());
    off += 6;
    // mov rdi, rax — the handle becomes exit's code
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]);
    off += 3;
    // call qword ptr [rip+disp32] -> SLOT_EXIT_OFF (never returns)
    let disp = (SLOT_EXIT_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());

    let path = b"/system/common/lib/libSceSysmodule.sprx\0";
    image[PATH_OFF..PATH_OFF + path.len()].copy_from_slice(path);

    let module = SprxModule {
        name: "lsmTestModule".to_string(),
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
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };
    let dynlib = DynlibData {
        symbols: vec![
            DynSymbol {
                nid: nid_of("sceKernelLoadStartModule"),
                value: 0,
                is_import: true,
            },
            DynSymbol {
                nid: nid_of("exit"),
                value: 0,
                is_import: true,
            },
        ],
        relocations: vec![
            SceRela {
                offset: SLOT_LSM_OFF as u64,
                info: R_X86_64_JUMP_SLOT, // r_sym = 0
                addend: 0,
            },
            SceRela {
                offset: SLOT_EXIT_OFF as u64,
                info: (1u64 << 32) | R_X86_64_JUMP_SLOT, // r_sym = 1
                addend: 0,
            },
        ],
        ..Default::default()
    };

    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("sceKernelLoadStartModule/exit imports must link");
    assert!(
        linked.unresolved.is_empty(),
        "both imports must resolve to HLE"
    );

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("LoadStartModule call must not fault");
    assert_eq!(
        outcome,
        RunOutcome::Exited(1),
        "the guest must receive handle 1 (first module on a fresh kernel) from sceKernelLoadStartModule"
    );
    assert!(
        kernel.find_module("libSceSysmodule").is_some(),
        "the pseudo-module must be registered in the kernel module table"
    );
}

// --- M1 HLE-breadth hardening: new libc string fns resolve + dispatch -----

/// Proves the M1 hardening batch (memcmp/strchr/... ported from SharpEmu +
/// Kyty references) resolves through the *real* LM1 linker and dispatches:
/// a `_start` calls `strchr("a/b", '/')` on an image-resident string and
/// exits with the returned pointer's low byte offset from the string base
/// (== 1). If the NID didn't resolve, linking would leave it unresolved and
/// the call would fault instead.
#[test]
fn new_libc_strchr_resolves_and_dispatches_through_the_linker() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_STRCHR_OFF: usize = 0x80;
    const SLOT_EXIT_OFF: usize = 0x88;
    const STR_OFF: usize = 0x90;
    const STR_BASE: u64 = GUEST_ARENA_BASE + STR_OFF as u64;

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;
    // lea rdi, [rip+disp32] -> STR_OFF
    let disp = (STR_OFF as i64 - (off as i64 + 7)) as i32;
    image[off..off + 3].copy_from_slice(&[0x48, 0x8D, 0x3D]);
    image[off + 3..off + 7].copy_from_slice(&disp.to_le_bytes());
    off += 7;
    // mov esi, '/'
    image[off] = 0xBE;
    image[off + 1..off + 5].copy_from_slice(&(b'/' as u32).to_le_bytes());
    off += 5;
    // call [rip -> SLOT_STRCHR_OFF]  (rax = guest ptr to '/')
    let disp = (SLOT_STRCHR_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());
    off += 6;
    // mov rdi, rax — exit with the full returned pointer
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]);
    off += 3;
    // call [rip -> SLOT_EXIT_OFF]
    let disp = (SLOT_EXIT_OFF as i64 - (off as i64 + 6)) as i32;
    image[off] = 0xFF;
    image[off + 1] = 0x15;
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());

    image[STR_OFF..STR_OFF + 4].copy_from_slice(b"a/b\0");

    let module = SprxModule {
        name: "strchrTest".to_string(),
        e_type: 0xFE18,
        segments: vec![SprxSegment {
            vaddr: 0,
            data: image,
            flags: 5,
            mem_size: 0x100,
        }],
        dynlib_data: None,
        relro: None,
        dynamic: None,
        entry: ENTRY_OFF as u64,
        tls: None,
        procparam: None,
    };
    let dynlib = DynlibData {
        symbols: vec![
            DynSymbol {
                nid: nid_of("strchr"),
                value: 0,
                is_import: true,
            },
            DynSymbol {
                nid: nid_of("exit"),
                value: 0,
                is_import: true,
            },
        ],
        relocations: vec![
            SceRela {
                offset: SLOT_STRCHR_OFF as u64,
                info: R_X86_64_JUMP_SLOT,
                addend: 0,
            },
            SceRela {
                offset: SLOT_EXIT_OFF as u64,
                info: (1u64 << 32) | R_X86_64_JUMP_SLOT,
                addend: 0,
            },
        ],
        ..Default::default()
    };

    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE).expect("links");
    assert!(
        linked.unresolved.is_empty(),
        "strchr and exit must resolve to HLE (the batch is registered)"
    );

    let kernel = OrbisKernel::new();
    let outcome =
        execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).expect("must not fault");
    assert_eq!(
        outcome,
        RunOutcome::Exited(STR_BASE + 1),
        "strchr must return the guest address of '/' — offset 1 of \"a/b\" at STR_BASE"
    );
}
