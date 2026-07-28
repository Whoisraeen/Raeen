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

use raeen_firmware::dynlib::nid::{NidDatabase, nid_of};
use raeen_firmware::dynlib::{DynSymbol, DynlibData, SceRela, SymbolRef};
use raeen_firmware::{
    HLE_TRAMPOLINE_BASE, HleTrampoline, LinkedModule, ModuleInit, ModuleInitRole, ModuleRegistry,
    SprxModule, SprxSegment, TlsTemplate, UNRESOLVED_STUB_BASE, UnresolvedStub, link_module,
};
use raeen_hle::{HleContext, HleRegistry};
use raeen_kernel::OrbisKernel;
use raeen_runtime::{
    GUEST_ARENA_BASE, RunOutcome, RuntimeError, execute_linked, execute_process,
    execute_process_shared, fsbase_rearm_count, hle_dispatch_metrics,
};

const R_X86_64_JUMP_SLOT: u64 = 7;

/// Forces real scheduler preemption of the guest thread by confining **both**
/// the current (guest-running) thread and one busy-spinner to a *single*
/// logical CPU via thread affinity — two runnable threads on one core, so the
/// scheduler must time-slice them (preemption guaranteed), while every other
/// core stays free.
///
/// This is deliberately NOT whole-machine saturation: the runtime test suite
/// runs in parallel, and a test that pegs every core stalls all the others
/// (and, if it then panics, leaks the spinners, wedging the whole binary and
/// even locking its `.exe` against relinking). Confining to one core keeps the
/// test a good citizen and makes preemption *more* reliable, not less. On
/// `Drop` (including on an assertion panic) it stops+joins the spinner and
/// restores the caller's original affinity.
struct Spinners {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    prev_affinity: usize,
}

impl Spinners {
    /// Pin the calling thread + one spinner to CPU 0.
    fn contend_one_core() -> Self {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
        // SAFETY: `GetCurrentThread` is a pseudo-handle; `SetThreadAffinityMask`
        // with mask 1 (CPU 0) is a benign scheduling hint that returns the
        // previous mask (0 on failure). Restored on Drop.
        let prev_affinity = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1) };
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let handle = {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                // SAFETY: pin *this* spinner to CPU 0 too, so it shares a core
                // with the guest thread.
                unsafe { SetThreadAffinityMask(GetCurrentThread(), 1) };
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    std::hint::spin_loop();
                }
            })
        };
        Self {
            stop,
            handle: Some(handle),
            prev_affinity,
        }
    }
}

impl Drop for Spinners {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        if self.prev_affinity != 0 {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
            // SAFETY: restore the caller's original affinity mask. Drop runs on
            // the same thread that constructed this (the test/guest thread).
            unsafe { SetThreadAffinityMask(GetCurrentThread(), self.prev_affinity) };
        }
    }
}

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

fn direct_strlen_module(iterations: u32) -> LinkedModule {
    const SLOT_OFF: usize = 0x80;
    const STRING_OFF: usize = 0x100;
    let mut image = vec![0u8; 0x180];
    let string_addr = GUEST_ARENA_BASE + STRING_OFF as u64;
    let mut off = 0usize;
    image[off..off + 2].copy_from_slice(&[0x41, 0xBC]); // mov r12d,imm32
    image[off + 2..off + 6].copy_from_slice(&iterations.to_le_bytes());
    off += 6;
    let loop_off = off;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBF]); // mov rdi,imm64
    image[off + 2..off + 10].copy_from_slice(&string_addr.to_le_bytes());
    off += 10;
    let next = off as i64 + 6;
    let disp = (SLOT_OFF as i64 - next) as i32;
    image[off..off + 2].copy_from_slice(&[0xFF, 0x15]);
    image[off + 2..off + 6].copy_from_slice(&disp.to_le_bytes());
    off += 6;
    image[off..off + 3].copy_from_slice(&[0x49, 0xFF, 0xCC]); // dec r12
    off += 3;
    image[off..off + 2].copy_from_slice(&[0x75, (loop_off as i8 - (off as i8 + 2)) as u8]);
    off += 2;
    image[off] = 0xC3;
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    image[STRING_OFF..STRING_OFF + 6].copy_from_slice(b"raeen\0");

    LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "strlen".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    }
}

fn direct_import_module(library: &str, function: &str) -> LinkedModule {
    const SLOT_OFF: usize = 0x80;
    let mut image = vec![0u8; 0x100];
    write_entry_stub(&mut image, 0, SLOT_OFF);
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: library.to_owned(),
            function: function.to_owned(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    }
}

#[test]
fn executable_leaf_thunk_dispatches_without_veh() {
    if !fsgsbase_available() {
        return;
    }
    let before = hle_dispatch_metrics();
    let value = execute_linked(
        &direct_strlen_module(1_000),
        &HleRegistry::new(),
        &OrbisKernel::new(),
        0,
        &[],
    )
    .expect("direct strlen loop must return");
    let after = hle_dispatch_metrics();
    assert_eq!(value, 5);
    // Dispatch counters are process-global and the integration binary runs
    // tests in parallel; another direct-enabled fixture can add calls during
    // this window. This guest itself contributes exactly `iterations`, so the
    // lower bound is the race-free assertion.
    assert!(after.direct - before.direct >= 1_000);
    // VEH is process-global and other integration tests execute in parallel,
    // so only the isolated benchmark below can assert an exact zero VEH delta.
    assert!(after.veh >= before.veh);
}

#[test]
fn executable_ordinary_thunk_dispatches_without_veh() {
    if !fsgsbase_available() || std::env::var_os("RAEEN_DISABLE_DIRECT_HLE").is_some() {
        return;
    }
    let before = hle_dispatch_metrics();
    let value = execute_linked(
        &direct_import_module("libkernel", "sceKernelGetProcessTimeCounterFrequency"),
        &HleRegistry::new(),
        &OrbisKernel::new(),
        0,
        &[],
    )
    .expect("ordinary direct HLE call must return");
    let after = hle_dispatch_metrics();
    assert_eq!(value, 1_000_000_000);
    // Process-global counters can include a concurrent integration test. The
    // classification unit test pins this function to the direct bridge; here
    // we prove that executing it contributes a direct dispatch.
    assert!(after.direct > before.direct);
    assert!(after.veh >= before.veh);
}

#[test]
#[ignore = "manual one-million-call performance benchmark"]
fn benchmark_one_million_executable_hle_calls() {
    if !fsgsbase_available() {
        return;
    }
    let before = hle_dispatch_metrics();
    let started = std::time::Instant::now();
    let value = execute_linked(
        &direct_strlen_module(1_000_000),
        &HleRegistry::new(),
        &OrbisKernel::new(),
        0,
        &[],
    )
    .expect("benchmark guest must return");
    let elapsed = started.elapsed();
    let after = hle_dispatch_metrics();
    assert_eq!(value, 5);
    let direct = after.direct - before.direct;
    let veh = after.veh - before.veh;
    if std::env::var_os("RAEEN_DISABLE_DIRECT_HLE").is_some() {
        assert_eq!(veh, 1_000_000);
        assert_eq!(direct, 0);
    } else {
        assert_eq!(direct, 1_000_000);
        assert_eq!(veh, 0);
    }
    eprintln!(
        "HLE_BENCH mode={} iterations=1000000 elapsed_ms={:.3} calls_per_second={:.0} veh={} direct={}",
        if direct == 0 { "veh" } else { "direct" },
        elapsed.as_secs_f64() * 1_000.0,
        1_000_000.0 / elapsed.as_secs_f64(),
        veh,
        direct,
    );
}

/// Add the provider tables a real SCE symbol carries. Runtime fixtures used
/// to omit them and relied on provider-free NID lookup, which cannot model two
/// libraries exporting an equal numeric NID.
fn bind_import_providers(mut dynlib: DynlibData, providers: &[&str]) -> DynlibData {
    assert_eq!(dynlib.symbols.len(), providers.len());
    let refs: Vec<SymbolRef> = dynlib
        .symbols
        .iter()
        .zip(providers)
        .enumerate()
        .map(|(index, (symbol, _))| SymbolRef {
            nid: symbol.nid,
            library_index: (index + 1) as u16,
            module_index: (index + 1) as u16,
        })
        .collect();
    dynlib.imports = refs.clone();
    dynlib.symbol_providers = refs.into_iter().map(Some).collect();
    dynlib.import_libs = providers
        .iter()
        .enumerate()
        .map(|(index, provider)| ((index + 1) as u16, (*provider).to_string()))
        .collect();
    dynlib.import_modules = dynlib.import_libs.clone();
    dynlib
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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libtest"],
    );

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

#[test]
fn native_guest_syscall_is_trapped_and_dispatched_as_orbis_not_windows() {
    // mov eax, SYS_getpid(20); syscall; ret
    let image = vec![0xB8, 20, 0, 0, 0, 0x0F, 0x05, 0xC3];
    let module = SprxModule {
        name: "orbisSyscallTest".to_string(),
        e_type: 0xFE18,
        segments: vec![SprxSegment {
            vaddr: 0,
            data: image,
            flags: 5,
            mem_size: 8,
        }],
        dynlib_data: None,
        relro: None,
        dynamic: None,
        entry: 0,
        tls: None,
        procparam: None,
        unwind: None,
    };
    let hle = HleRegistry::new();
    let registry = ModuleRegistry::new(NidDatabase::from_hle_names(hle.registered_names()));
    let linked = link_module(
        &module,
        &DynlibData::default(),
        &registry,
        &hle,
        GUEST_ARENA_BASE,
    )
    .expect("syscall fixture links");
    assert_eq!(
        &linked.image[5..7],
        &raeen_firmware::dynlib::linker::SYSCALL_TRAP_BYTES,
        "the Windows SYSCALL instruction must be gone before native execution"
    );

    let result = execute_linked(&linked, &hle, &OrbisKernel::new(), 0, &[])
        .expect("the private trap dispatches through the Orbis kernel");
    assert_eq!(result, 1, "the emulated process has PID 1");
}

/// A guest callback requested by an HLE function must execute in the active
/// guest context and return to the instruction after the original import
/// call. `pthread_once` is the first consumer, but the context-transfer
/// mechanism is intentionally generic for callbacks used by other PS5 APIs.
#[test]
fn pthread_once_runs_its_guest_initializer_before_returning() {
    const ENTRY_OFF: usize = 0x00;
    const INIT_OFF: usize = 0x40;
    const SLOT_OFF: usize = 0x70;
    const ONCE_OFF: usize = 0x80;
    const RESULT_OFF: usize = 0x88;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];

    // lea rdi, [rip + once_control]
    image[0x00..0x07].copy_from_slice(&[0x48, 0x8D, 0x3D, 0x79, 0x00, 0x00, 0x00]);
    // lea rsi, [rip + initializer]
    image[0x07..0x0E].copy_from_slice(&[0x48, 0x8D, 0x35, 0x32, 0x00, 0x00, 0x00]);
    // call qword ptr [rip + import_slot]
    image[0x0E..0x14].copy_from_slice(&[0xFF, 0x15, 0x5C, 0x00, 0x00, 0x00]);
    // Call the same once-control again. Its initializer must not repeat.
    image[0x14..0x1A].copy_from_slice(&[0xFF, 0x15, 0x56, 0x00, 0x00, 0x00]);
    // mov eax, dword ptr [rip + result]; ret
    image[0x1A..0x21].copy_from_slice(&[0x8B, 0x05, 0x68, 0x00, 0x00, 0x00, 0xC3]);

    // initializer: inc dword ptr [rip + result]; ret
    image[INIT_OFF..INIT_OFF + 7].copy_from_slice(&[0xFF, 0x05, 0x42, 0x00, 0x00, 0x00, 0xC3]);
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libkernel".to_string(),
            function: "scePthreadOnce".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[])
        .expect("pthread_once guest initializer must execute and return");
    assert_eq!(result, 1, "the once initializer must run exactly once");

    // Keep the offsets used above honest if the fixture is edited.
    assert_eq!(ONCE_OFF, 0x80);
    assert_eq!(RESULT_OFF, 0x88);
}

/// `sceKernelRaiseException` acceptance test: a guest that installs an
/// exception handler and raises `SIGUSR1` at itself must have that **guest
/// handler actually execute**, with the FreeBSD signal ABI, before the raising
/// code observes its effect.
///
/// # What this replaces
///
/// `Raise` used to log `"guest handler is registered but asynchronous delivery
/// is not implemented; acknowledging"` and return `SCE_OK`. That is the measured
/// first blocker for Subnautica Below Zero, which timed out at 180 s having
/// burned 1.4 s of CPU and produced zero flips: `SIGUSR1` (30) is what a managed
/// runtime's stop-the-world collector raises to suspend a thread, so
/// acknowledging without delivering leaves the collector waiting forever for a
/// suspension that never happens.
///
/// # The fixture
///
/// Entry: `mov edi,30; lea rsi,[handler]; call [InstallExceptionHandler]` then
/// `mov edi,1 (self); mov esi,30; call [RaiseException]`, and finally returns
/// the marker word the handler is supposed to have written.
///
/// Handler (real guest code, entered through the runtime's synchronous
/// guest-callback path): starts from its `signum` argument, then *validates the
/// machine context it was handed* — `or eax,0x100` only if `uctx->uc_mcontext
/// .mc_len == sizeof(mcontext_t)`, and `or eax,0x200` only if `mc_rip` is
/// non-zero — before storing the result.
///
/// So the single returned value proves four things at once: the handler ran, it
/// received the right signal number, `arg1` is a real `ucontext_t` and not a
/// stray pointer, and it describes the interrupted guest instruction.
#[test]
fn raise_exception_runs_the_installed_guest_handler_with_a_real_ucontext() {
    const HANDLER_OFF: usize = 0x80;
    const MARKER_OFF: usize = 0x100;
    const SLOT_INSTALL: usize = 0x110;
    const SLOT_RAISE: usize = 0x118;
    /// `offsetof(ucontext_t, uc_mcontext.mc_len)` — 0x40 + 0xC8.
    const MC_LEN_AT: u32 = 0x108;
    /// `offsetof(ucontext_t, uc_mcontext.mc_rip)` — 0x40 + 0xA0.
    const MC_RIP_AT: u32 = 0xE0;
    /// `sizeof(mcontext_t)` the handler checks `mc_len` against.
    const MCONTEXT_LEN: u32 = 0x480;
    const SIGUSR1: u32 = 30;
    /// Set by the handler when `mc_len` was correct.
    const SAW_MCONTEXT_LEN: u64 = 0x100;
    /// Set by the handler when `mc_rip` named a real instruction.
    const SAW_MCONTEXT_RIP: u64 = 0x200;

    let mut image = vec![0u8; 0x200];

    // ---- entry ---------------------------------------------------------
    // mov edi, 30                            (signum)
    image[0x00..0x05].copy_from_slice(&[0xBF, 0x1E, 0x00, 0x00, 0x00]);
    // lea rsi, [rip + handler]
    image[0x05..0x08].copy_from_slice(&[0x48, 0x8D, 0x35]);
    image[0x08..0x0C].copy_from_slice(&((HANDLER_OFF as i32) - 0x0C).to_le_bytes());
    // call qword ptr [rip + install_slot]
    image[0x0C..0x0E].copy_from_slice(&[0xFF, 0x15]);
    image[0x0E..0x12].copy_from_slice(&((SLOT_INSTALL as i32) - 0x12).to_le_bytes());
    // mov edi, 1                             (target thread = the main thread)
    image[0x12..0x17].copy_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]);
    // mov esi, 30                            (signum)
    image[0x17..0x1C].copy_from_slice(&[0xBE, 0x1E, 0x00, 0x00, 0x00]);
    // call qword ptr [rip + raise_slot]
    image[0x1C..0x1E].copy_from_slice(&[0xFF, 0x15]);
    image[0x1E..0x22].copy_from_slice(&((SLOT_RAISE as i32) - 0x22).to_le_bytes());
    // mov eax, dword ptr [rip + marker] ; ret
    image[0x22..0x24].copy_from_slice(&[0x8B, 0x05]);
    image[0x24..0x28].copy_from_slice(&((MARKER_OFF as i32) - 0x28).to_le_bytes());
    image[0x28] = 0xC3;

    // ---- handler: void handler(int signum, ucontext_t *uctx) -----------
    let mut off = HANDLER_OFF;
    // mov eax, edi                           (start from the delivered signum)
    image[off..off + 2].copy_from_slice(&[0x89, 0xF8]);
    off += 2;
    // mov rdx, qword ptr [rsi + mc_len]
    image[off..off + 3].copy_from_slice(&[0x48, 0x8B, 0x96]);
    image[off + 3..off + 7].copy_from_slice(&MC_LEN_AT.to_le_bytes());
    off += 7;
    // cmp rdx, sizeof(mcontext_t)
    image[off..off + 3].copy_from_slice(&[0x48, 0x81, 0xFA]);
    image[off + 3..off + 7].copy_from_slice(&MCONTEXT_LEN.to_le_bytes());
    off += 7;
    // jne +5   (skip the `or`)
    image[off..off + 2].copy_from_slice(&[0x75, 0x05]);
    off += 2;
    // or eax, SAW_MCONTEXT_LEN
    image[off] = 0x0D;
    image[off + 1..off + 5].copy_from_slice(&(SAW_MCONTEXT_LEN as u32).to_le_bytes());
    off += 5;
    // mov rcx, qword ptr [rsi + mc_rip]
    image[off..off + 3].copy_from_slice(&[0x48, 0x8B, 0x8E]);
    image[off + 3..off + 7].copy_from_slice(&MC_RIP_AT.to_le_bytes());
    off += 7;
    // test rcx, rcx ; je +5
    image[off..off + 5].copy_from_slice(&[0x48, 0x85, 0xC9, 0x74, 0x05]);
    off += 5;
    // or eax, SAW_MCONTEXT_RIP
    image[off] = 0x0D;
    image[off + 1..off + 5].copy_from_slice(&(SAW_MCONTEXT_RIP as u32).to_le_bytes());
    off += 5;
    // mov dword ptr [rip + marker], eax ; ret
    image[off..off + 2].copy_from_slice(&[0x89, 0x05]);
    let after = off as i32 + 6;
    image[off + 2..off + 6].copy_from_slice(&((MARKER_OFF as i32) - after).to_le_bytes());
    off += 6;
    image[off] = 0xC3;
    assert!(
        off < MARKER_OFF,
        "the handler must not overlap the marker word"
    );

    image[SLOT_INSTALL..SLOT_INSTALL + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    image[SLOT_RAISE..SLOT_RAISE + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + 8).to_le_bytes());

    let hle = HleRegistry::new();
    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![
            HleTrampoline {
                library: "libkernel".to_string(),
                function: "sceKernelInstallExceptionHandler".to_string(),
                addr: HLE_TRAMPOLINE_BASE,
            },
            HleTrampoline {
                library: "libkernel".to_string(),
                function: "sceKernelRaiseException".to_string(),
                addr: HLE_TRAMPOLINE_BASE + 8,
            },
        ],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let before = raeen_hle::exception::delivered_count();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("the raise and its handler must both complete");

    assert_ne!(
        result, 0,
        "the guest handler never ran: `sceKernelRaiseException` acknowledged without delivering"
    );
    assert_eq!(
        result & 0xFF,
        u64::from(SIGUSR1),
        "the handler must receive the raised Orbis signal number as arg0"
    );
    assert_eq!(
        result & SAW_MCONTEXT_LEN,
        SAW_MCONTEXT_LEN,
        "arg1 must point at a real ucontext_t: mc_len must be sizeof(mcontext_t)"
    );
    assert_eq!(
        result & SAW_MCONTEXT_RIP,
        SAW_MCONTEXT_RIP,
        "the machine context must name the interrupted guest instruction (mc_rip != 0)"
    );

    assert_eq!(
        raeen_hle::exception::delivered_count(),
        before + 1,
        "exactly one delivery must be counted"
    );
    assert!(
        kernel.pending_exceptions.is_empty(),
        "a delivered exception must not stay queued"
    );
    assert!(
        kernel.exception_delivery_active.is_empty(),
        "the delivering mark must be released"
    );
}

fn emit_indirect_call(image: &mut [u8], off: &mut usize, slot: usize) {
    let after = *off as i64 + 6;
    let disp = (slot as i64 - after) as i32;
    image[*off..*off + 2].copy_from_slice(&[0xFF, 0x15]);
    image[*off + 2..*off + 6].copy_from_slice(&disp.to_le_bytes());
    *off += 6;
}

/// `__stack_chk_fail` is `noreturn` on hardware: the compiler emits it as the
/// last instruction of a smashed epilogue, so an HLE handler that *returns*
/// makes the guest execute whatever bytes follow the call. The poison tail
/// (`mov eax, POISON; ret`) sits immediately after the call — the old
/// return-0 stub surfaced `POISON`, exactly the walk-into-UD2 that masked
/// Until Dawn's real canary smash. The fixed handler must unwind the guest
/// thread with the fatal exit code instead.
#[test]
fn stack_chk_fail_unwinds_the_guest_instead_of_returning() {
    const SLOT: usize = 0x40;
    const POISON: u32 = 0xBAD;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x80];
    let mut off = 0usize;
    emit_indirect_call(&mut image, &mut off, SLOT); // call __stack_chk_fail
    image[off] = 0xB8; // mov eax, POISON  (must never execute)
    image[off + 1..off + 5].copy_from_slice(&POISON.to_le_bytes());
    off += 5;
    image[off] = 0xC3; // ret
    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "__stack_chk_fail".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("a canary smash is a reported guest-fatal unwind, not a host fault");
    assert_ne!(
        result,
        u64::from(POISON),
        "__stack_chk_fail returned control to the smashed guest frame"
    );
    assert_eq!(
        result,
        raeen_hle::STACK_CHK_FAIL_EXIT_CODE,
        "the guest thread must unwind with the stack-smash exit code"
    );
}

/// `abort()` is `noreturn` on hardware: a title's fatal path (assert,
/// terminate, panic handler) ends in `call abort` with garbage — often UD2 —
/// after it. The old stub returned 0, walking the guest into that garbage
/// and misreporting the deliberate abort as a wild fault. The fixed handler
/// must unwind the guest thread with the abort fatal code, exactly like the
/// `__stack_chk_fail` test above (same walk-into-poison layout).
#[test]
fn abort_unwinds_the_guest_instead_of_returning() {
    const SLOT: usize = 0x40;
    const POISON: u32 = 0xBAD;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x80];
    let mut off = 0usize;
    emit_indirect_call(&mut image, &mut off, SLOT); // call abort
    image[off] = 0xB8; // mov eax, POISON  (must never execute)
    image[off + 1..off + 5].copy_from_slice(&POISON.to_le_bytes());
    off += 5;
    image[off] = 0xC3; // ret
    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "abort".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("a guest abort is a reported guest-fatal unwind, not a host fault");
    assert_ne!(
        result,
        u64::from(POISON),
        "abort returned control to the guest frame that called it"
    );
    assert_eq!(
        result,
        raeen_hle::ABORT_EXIT_CODE,
        "the guest thread must unwind with the abort exit code"
    );
}

/// `exit(status)` is `noreturn` on hardware: nothing after the call site may
/// execute, and the status must be the run's outcome. Same poison-tail
/// layout as the abort/stack-chk tests — an exit that merely returned to its
/// caller would surface as `POISON` instead of the status.
#[test]
fn exit_unwinds_the_guest_with_its_status_instead_of_returning() {
    const SLOT: usize = 0x40;
    const POISON: u32 = 0xBAD;
    const STATUS: u8 = 0x2A;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x80];
    let mut off = 0usize;
    // mov edi, STATUS  (SysV arg0 = exit status)
    image[off] = 0xBF;
    image[off + 1..off + 5].copy_from_slice(&u32::from(STATUS).to_le_bytes());
    off += 5;
    emit_indirect_call(&mut image, &mut off, SLOT); // call exit
    image[off] = 0xB8; // mov eax, POISON  (must never execute)
    image[off + 1..off + 5].copy_from_slice(&POISON.to_le_bytes());
    off += 5;
    image[off] = 0xC3; // ret
    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("exit() must terminate the run cleanly, not fault");
    assert_ne!(
        result,
        u64::from(POISON),
        "exit returned control to the guest frame that called it"
    );
    assert_eq!(
        result,
        u64::from(STATUS),
        "the run must end with the guest's own exit status"
    );
}

/// M1-E: a worker that ends via `scePthreadExit(v)` — rather than returning —
/// unwinds its guest context and hands `v` to the joiner. The poison tail
/// (`mov eax, POISON; ret`) sits immediately after the exit call, so an exit
/// that merely returned to its caller would surface as `POISON` instead of
/// `MAGIC`. That is what makes this falsifiable rather than decorative.
#[test]
fn pthread_exit_unwinds_the_worker_and_delivers_its_value_to_join() {
    const WORKER: usize = 0x100;
    const THREAD_OUT: usize = 0x180;
    const RETVAL_OUT: usize = 0x188;
    const CREATE_SLOT: usize = 0x1C0;
    const JOIN_SLOT: usize = 0x1C8;
    const EXIT_SLOT: usize = 0x1D0;
    const PTHREAD_EXIT_SLOT: usize = 0x1D8;
    const MAGIC: u32 = 0x5A;
    const POISON: u32 = 0xBAD;

    let hle = std::sync::Arc::new(HleRegistry::new());
    let kernel = std::sync::Arc::new(OrbisKernel::new());
    let mut image = vec![0u8; 0x300];

    let thread_out = GUEST_ARENA_BASE + THREAD_OUT as u64;
    let retval_out = GUEST_ARENA_BASE + RETVAL_OUT as u64;
    let worker = GUEST_ARENA_BASE + WORKER as u64;

    // main: create(worker) -> join(thread, &retval) -> exit(retval)
    let mut off = 0;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBF]); // mov rdi, thread_out
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 2].copy_from_slice(&[0x31, 0xF6]); // xor esi, esi (attr = 0)
    off += 2;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBA]); // mov rdx, worker
    image[off + 2..off + 10].copy_from_slice(&worker.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x31, 0xC9]); // xor rcx, rcx (arg = 0)
    off += 3;
    emit_indirect_call(&mut image, &mut off, CREATE_SLOT);
    image[off..off + 2].copy_from_slice(&[0x48, 0xA1]); // mov rax, [thread_out]
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
    off += 3;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBE]); // mov rsi, retval_out
    image[off + 2..off + 10].copy_from_slice(&retval_out.to_le_bytes());
    off += 10;
    emit_indirect_call(&mut image, &mut off, JOIN_SLOT);
    image[off..off + 2].copy_from_slice(&[0x48, 0xA1]); // mov rax, [retval_out]
    image[off + 2..off + 10].copy_from_slice(&retval_out.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
    off += 3;
    emit_indirect_call(&mut image, &mut off, EXIT_SLOT);

    // worker: scePthreadExit(MAGIC), then unreachable poison.
    off = WORKER;
    image[off] = 0xBF; // mov edi, MAGIC
    image[off + 1..off + 5].copy_from_slice(&MAGIC.to_le_bytes());
    off += 5;
    emit_indirect_call(&mut image, &mut off, PTHREAD_EXIT_SLOT);
    image[off] = 0xB8; // mov eax, POISON  (must never execute)
    image[off + 1..off + 5].copy_from_slice(&POISON.to_le_bytes());
    off += 5;
    image[off] = 0xC3; // ret

    for (slot, index) in [
        (CREATE_SLOT, 0u64),
        (JOIN_SLOT, 1),
        (EXIT_SLOT, 2),
        (PTHREAD_EXIT_SLOT, 3),
    ] {
        image[slot..slot + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + index * 8).to_le_bytes());
    }
    let linked = std::sync::Arc::new(LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadCreate".into(),
                addr: HLE_TRAMPOLINE_BASE,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadJoin".into(),
                addr: HLE_TRAMPOLINE_BASE + 8,
            },
            HleTrampoline {
                library: "libc".into(),
                function: "exit".into(),
                addr: HLE_TRAMPOLINE_BASE + 16,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadExit".into(),
                addr: HLE_TRAMPOLINE_BASE + 24,
            },
        ],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    });

    let outcome = execute_process_shared(linked, hle, kernel, &["/app0/eboot.bin"], &[])
        .expect("a worker ending via scePthreadExit must still be joinable");
    assert_eq!(
        outcome,
        RunOutcome::Exited(MAGIC as u64),
        "join must observe the scePthreadExit value, not the poison tail"
    );
}

/// M1-E runtime scaffold: `scePthreadCreate` launches a genuinely
/// distinct OS thread with its own TCB, the worker survives a forced sleep,
/// reads `fs:0x28`, reports its distinct guest handle, and join returns that
/// value to the main guest context.
#[test]
fn pthread_create_join_runs_a_real_guest_worker_with_tls() {
    const WORKER: usize = 0x100;
    const THREAD_OUT: usize = 0x180;
    const RETVAL_OUT: usize = 0x188;
    const CREATE_SLOT: usize = 0x1C0;
    const JOIN_SLOT: usize = 0x1C8;
    const EXIT_SLOT: usize = 0x1D0;
    const USLEEP_SLOT: usize = 0x1D8;
    const GETTID_SLOT: usize = 0x1E0;

    let hle = std::sync::Arc::new(HleRegistry::new());
    let kernel = std::sync::Arc::new(OrbisKernel::new());
    let mut image = vec![0u8; 0x300];

    let thread_out = GUEST_ARENA_BASE + THREAD_OUT as u64;
    let retval_out = GUEST_ARENA_BASE + RETVAL_OUT as u64;
    let worker = GUEST_ARENA_BASE + WORKER as u64;
    let mut off = 0;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBF]); // mov rdi, thread_out
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 2].copy_from_slice(&[0x31, 0xF6]); // xor esi, esi
    off += 2;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBA]); // mov rdx, worker
    image[off + 2..off + 10].copy_from_slice(&worker.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x31, 0xC9]); // xor rcx, rcx
    off += 3;
    emit_indirect_call(&mut image, &mut off, CREATE_SLOT);
    image[off..off + 2].copy_from_slice(&[0x48, 0xA1]); // mov rax, [thread_out]
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
    off += 3;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBE]); // mov rsi, retval_out
    image[off + 2..off + 10].copy_from_slice(&retval_out.to_le_bytes());
    off += 10;
    emit_indirect_call(&mut image, &mut off, JOIN_SLOT);
    image[off..off + 2].copy_from_slice(&[0x48, 0xA1]); // mov rax, [retval_out]
    image[off + 2..off + 10].copy_from_slice(&retval_out.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
    off += 3;
    emit_indirect_call(&mut image, &mut off, EXIT_SLOT);

    off = WORKER;
    image[off] = 0xBF; // mov edi, 20000
    image[off + 1..off + 5].copy_from_slice(&20_000u32.to_le_bytes());
    off += 5;
    emit_indirect_call(&mut image, &mut off, USLEEP_SLOT);
    image[off..off + 9].copy_from_slice(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0, 0, 0]); // mov rax, fs:[0x28]
    off += 9;
    emit_indirect_call(&mut image, &mut off, GETTID_SLOT);
    image[off] = 0xC3;

    for (slot, index) in [
        (CREATE_SLOT, 0u64),
        (JOIN_SLOT, 1),
        (EXIT_SLOT, 2),
        (USLEEP_SLOT, 3),
        (GETTID_SLOT, 4),
    ] {
        image[slot..slot + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + index * 8).to_le_bytes());
    }
    let linked = std::sync::Arc::new(LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadCreate".into(),
                addr: HLE_TRAMPOLINE_BASE,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadJoin".into(),
                addr: HLE_TRAMPOLINE_BASE + 8,
            },
            HleTrampoline {
                library: "libc".into(),
                function: "exit".into(),
                addr: HLE_TRAMPOLINE_BASE + 16,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "sceKernelUsleep".into(),
                addr: HLE_TRAMPOLINE_BASE + 24,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadGetthreadid".into(),
                addr: HLE_TRAMPOLINE_BASE + 32,
            },
        ],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    });

    let outcome = execute_process_shared(linked, hle, kernel, &["/app0/eboot.bin"], &[])
        .expect("real guest worker create/join must complete");
    assert_eq!(outcome, RunOutcome::Exited(2));
}

#[test]
fn detached_worker_is_reaped_before_the_fixed_guest_arena_is_reused() {
    const WORKER: usize = 0x100;
    const THREAD_OUT: usize = 0x180;
    const CREATE_SLOT: usize = 0x1C0;
    const DETACH_SLOT: usize = 0x1C8;
    const EXIT_SLOT: usize = 0x1D0;
    const USLEEP_SLOT: usize = 0x1D8;

    let hle = std::sync::Arc::new(HleRegistry::new());
    let kernel = std::sync::Arc::new(OrbisKernel::new());
    let mut image = vec![0u8; 0x280];
    let thread_out = GUEST_ARENA_BASE + THREAD_OUT as u64;
    let worker = GUEST_ARENA_BASE + WORKER as u64;
    let mut off = 0;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBF]);
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 2].copy_from_slice(&[0x31, 0xF6]);
    off += 2;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBA]);
    image[off + 2..off + 10].copy_from_slice(&worker.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x31, 0xC9]);
    off += 3;
    emit_indirect_call(&mut image, &mut off, CREATE_SLOT);
    image[off..off + 2].copy_from_slice(&[0x48, 0xA1]);
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]);
    off += 3;
    emit_indirect_call(&mut image, &mut off, DETACH_SLOT);
    image[off..off + 2].copy_from_slice(&[0x31, 0xFF]);
    off += 2;
    emit_indirect_call(&mut image, &mut off, EXIT_SLOT);

    off = WORKER;
    image[off] = 0xBF;
    image[off + 1..off + 5].copy_from_slice(&20_000u32.to_le_bytes());
    off += 5;
    emit_indirect_call(&mut image, &mut off, USLEEP_SLOT);
    image[off] = 0xC3;

    for (slot, index) in [
        (CREATE_SLOT, 0u64),
        (DETACH_SLOT, 1),
        (EXIT_SLOT, 2),
        (USLEEP_SLOT, 3),
    ] {
        image[slot..slot + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + index * 8).to_le_bytes());
    }
    let linked = std::sync::Arc::new(LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadCreate".into(),
                addr: HLE_TRAMPOLINE_BASE,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadDetach".into(),
                addr: HLE_TRAMPOLINE_BASE + 8,
            },
            HleTrampoline {
                library: "libc".into(),
                function: "exit".into(),
                addr: HLE_TRAMPOLINE_BASE + 16,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "sceKernelUsleep".into(),
                addr: HLE_TRAMPOLINE_BASE + 24,
            },
        ],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    });

    for run in 0..2 {
        let outcome = execute_process_shared(
            std::sync::Arc::clone(&linked),
            std::sync::Arc::clone(&hle),
            std::sync::Arc::clone(&kernel),
            &["/app0/eboot.bin"],
            &[],
        )
        .unwrap_or_else(|error| panic!("run {run} must safely reuse fixed mappings: {error}"));
        assert_eq!(outcome, RunOutcome::Exited(0));
    }
}

fn unresolved_stub_module() -> (LinkedModule, u64, u64) {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;
    const WANTED_NID: u64 = 0x6F34_04C7_2D7C_F592;

    let mut image = vec![0u8; 0x100];
    write_entry_stub(&mut image, ENTRY_OFF, SLOT_OFF);
    // Index 2's stub — deliberately NOT index 0, so a regression that reports
    // the base address (or always picks slot 0) fails here.
    let stub_addr = UNRESOLVED_STUB_BASE + 2 * 8;
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&stub_addr.to_le_bytes());

    let stubs = vec![
        UnresolvedStub {
            nid: 0x1111,
            library: Some("libkernel".to_string()),
            addr: UNRESOLVED_STUB_BASE,
        },
        UnresolvedStub {
            nid: 0x2222,
            library: Some("libSceAgc".to_string()),
            addr: UNRESOLVED_STUB_BASE + 8,
        },
        UnresolvedStub {
            nid: WANTED_NID,
            library: Some("libc".to_string()),
            addr: stub_addr,
        },
    ];

    (
        LinkedModule {
            image,
            base: GUEST_ARENA_BASE,
            executable_ranges: Vec::new(),
            unresolved: Vec::new(),
            unresolved_stubs: stubs,
            module_inits: Vec::new(),
            hle_trampolines: Vec::<HleTrampoline>::new(),
            entry: ENTRY_OFF as u64,
            tls: None,
            tls_layout: Vec::new(),
            procparam_offset: None,
            unwind_modules: Vec::new(),
        },
        WANTED_NID,
        stub_addr,
    )
}

/// A guest `call` to a per-NID unresolved stub is a compatibility inventory
/// event by default: it resumes at the caller with `rax = 0`.
///
/// Here the guest calls table index 2 rather than the base slot. The kernel
/// inventory test separately pins NID/library/calling-module deduplication;
/// this native fixture proves the runtime performs the call/return stack
/// repair instead of terminating at the sentinel.
#[test]
fn call_to_unresolved_stub_resumes_with_zero_by_default() {
    let (linked, _, _) = unresolved_stub_module();
    let value = execute_linked(&linked, &HleRegistry::new(), &OrbisKernel::new(), 0, &[])
        .expect("default unresolved-call policy must resume");
    assert_eq!(value, 0, "unresolved call must synthesize rax=0");
}

/// The strict switch is validated in a child test process so its process-wide
/// OnceLock and environment cannot race the default-policy tests.
#[test]
fn strict_nids_restores_named_hard_failure() {
    const CHILD: &str = "RAEEN_STRICT_NIDS_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(
            std::env::current_exe().expect("integration-test executable path"),
        )
        .args([
            "--exact",
            "strict_nids_restores_named_hard_failure",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("RAEEN_STRICT_NIDS", "1")
        .status()
        .expect("launch isolated strict-NID test process");
        assert!(status.success(), "strict-NID child failed: {status}");
        return;
    }

    let (linked, wanted_nid, stub_addr) = unresolved_stub_module();
    let err = execute_linked(&linked, &HleRegistry::new(), &OrbisKernel::new(), 0, &[])
        .expect_err("strict mode must retain the named hard failure");
    assert_eq!(
        err,
        RuntimeError::UnimplementedImport {
            nid: wanted_nid,
            library: Some("libc".to_owned()),
            stub_addr,
            rip: stub_addr,
        }
    );
}

/// A guest fault that lands in the unresolved-stub *range* but names no
/// table entry is an ordinary wild fault, not an import — the reverse map
/// must not invent a NID for it.
#[test]
fn wild_fault_past_the_stub_table_is_still_an_anonymous_fault() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_OFF: usize = 0x10;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    write_entry_stub(&mut image, ENTRY_OFF, SLOT_OFF);
    // Slot 5, but the table below has only one entry.
    let wild = UNRESOLVED_STUB_BASE + 5 * 8;
    image[SLOT_OFF..SLOT_OFF + 8].copy_from_slice(&wild.to_le_bytes());

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        module_inits: Vec::new(),
        unresolved_stubs: vec![UnresolvedStub {
            nid: 0x1111,
            library: Some("libc".to_string()),
            addr: UNRESOLVED_STUB_BASE,
        }],
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    assert!(
        matches!(err, RuntimeError::Faulted { addr, .. } if addr == wild),
        "an in-range address with no table entry is an anonymous fault, not an \
         invented NID; got {err:?}"
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let args = [1u64, 2, 3, 4, 5, 6, 7];
    let err = execute_linked(&linked, &hle, &kernel, 0, &args).unwrap_err();
    assert_eq!(err, RuntimeError::TooManyArgs);
}

/// A guest `div` by zero is an x86 divide-error trap (`#DE`), which Windows
/// delivers as `EXCEPTION_INT_DIVIDE_BY_ZERO` (`0xC000_0094`) — **not** an
/// access violation. The VEH used to service only access violations, illegal
/// instructions and breakpoints and pass everything else on with
/// `EXCEPTION_CONTINUE_SEARCH`, so with no other handler installed this killed
/// the entire process with exit code `0xC000_0094` and *no log line at all*.
///
/// That is the measured A Plague Tale Requiem signature: `crashed` at 40.8 s,
/// exit `-1073741676` (= `0xC000_0094`), zero flips, zero unresolved NIDs, and
/// no ERROR line before it. Unhandled meant invisible by construction.
///
/// The entry is `xor ecx,ecx; xor edx,edx; mov eax,1; div ecx; ret`, so the
/// divisor is provably zero and the faulting instruction's address is known
/// exactly. The run must come back as `Err(IntegerDivideFault { .. })` naming
/// the `div`, and the test process must survive to run more guest code.
#[test]
fn guest_divide_by_zero_is_classified_instead_of_killing_the_process() {
    const ENTRY_OFF: usize = 0x0;
    // xor ecx,ecx (2) | xor edx,edx (2) | mov eax,1 (5) | div ecx (2) | ret (1)
    const DIVIDE_AT: u64 = 2 + 2 + 5;
    #[rustfmt::skip]
    const CODE: [u8; 12] = [
        0x31, 0xC9,                         // xor ecx, ecx   -> divisor = 0
        0x31, 0xD2,                         // xor edx, edx   -> clear dividend high
        0xB8, 0x01, 0x00, 0x00, 0x00,       // mov eax, 1
        0xF7, 0xF1,                         // div ecx        -> #DE
        0xC3,                               // ret            (never reached)
    ];

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    image[ENTRY_OFF..ENTRY_OFF + CODE.len()].copy_from_slice(&CODE);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    match err {
        RuntimeError::IntegerDivideFault {
            rip,
            cause,
            origin,
            hle: _,
        } => {
            assert_eq!(
                rip,
                GUEST_ARENA_BASE + ENTRY_OFF as u64 + DIVIDE_AT,
                "the report must name the faulting `div`, not the entry or the return"
            );
            assert_eq!(cause, raeen_runtime::DivideFault::ByZero);
            assert_eq!(
                origin,
                raeen_runtime::FaultOrigin::Guest,
                "a fault at a guest-arena address is the title's instruction, not ours"
            );
        }
        other => panic!("expected Err(IntegerDivideFault {{ .. }}), got {other:?}"),
    }

    // The process survived and the VEH/ACTIVE_CONTEXT state was fully torn down
    // and re-armed: prove it by running ordinary guest code on this same thread
    // right after the recovered trap.
    let mut ok_image = vec![0u8; 0x100];
    // mov eax, 7 ; ret
    ok_image[..6].copy_from_slice(&[0xB8, 0x07, 0x00, 0x00, 0x00, 0xC3]);
    let ok = LinkedModule {
        image: ok_image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };
    assert_eq!(
        execute_linked(&ok, &HleRegistry::new(), &OrbisKernel::new(), 0, &[]).unwrap(),
        7,
        "the runtime must still be usable after recovering a divide-error trap"
    );
}

/// The other `#DE` cause: `idiv` with `INT_MIN / -1`, whose quotient does not
/// fit the destination. Windows reports it as `EXCEPTION_INT_OVERFLOW`
/// (`0xC000_0095`) — a different code but the same instruction and the same
/// silent-process-death signature, so it must classify too rather than fall
/// through to `EXCEPTION_CONTINUE_SEARCH`.
#[test]
fn guest_idiv_quotient_overflow_is_classified_as_a_divide_fault() {
    // mov eax,imm32 (5) | cdq (1) | mov ecx,imm32 (5)
    const DIVIDE_AT: u64 = 5 + 1 + 5;
    #[rustfmt::skip]
    const CODE: [u8; 14] = [
        0xB8, 0x00, 0x00, 0x00, 0x80,       // mov eax, 0x80000000  (INT_MIN)
        0x99,                               // cdq                  (sign-extend)
        0xB9, 0xFF, 0xFF, 0xFF, 0xFF,       // mov ecx, -1
        0xF7, 0xF9,                         // idiv ecx             -> #DE
        0xC3,                               // ret                  (never reached)
    ];
    let mut image = vec![0u8; 0x100];
    image[..CODE.len()].copy_from_slice(&CODE);
    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &HleRegistry::new(), &kernel, 0, &[]).unwrap_err();
    match err {
        RuntimeError::IntegerDivideFault {
            rip, cause, origin, ..
        } => {
            assert_eq!(
                rip,
                GUEST_ARENA_BASE + DIVIDE_AT,
                "the report must name the faulting `idiv`"
            );
            assert_eq!(cause, raeen_runtime::DivideFault::Overflow);
            assert_eq!(origin, raeen_runtime::FaultOrigin::Guest);
        }
        other => panic!("expected Err(IntegerDivideFault {{ .. }}), got {other:?}"),
    }
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr, access, kind } => {
            // `addr` is the *faulting instruction's* Rip (the mapped
            // entry's host address), not the wild pointer (`0`) it
            // dereferenced.
            assert_ne!(
                addr, 0,
                "Faulted::addr is the faulting Rip, which is a real mapped-image address"
            );
            // ...and `access` is the wild pointer itself. The two together are
            // what make a fault diagnosable: Rip says where the guest was,
            // `access` says what it touched.
            assert_eq!(access, 0, "the guest dereferenced address 0");
            assert_eq!(kind, raeen_runtime::FaultKind::Read);
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
/// genuine VEH trap-and-dispatch path. `memcpy`'s [`raeen_hle::HleContext`]
/// gives it a [`raeen_hle::GuestMemory`] view of this same mapped image
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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libc"],
    );

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
    // crate-external file, with no access to `raeen_runtime::arena`) so the
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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libc"],
    );

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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libc", "libc"],
    );

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
/// region from the console's kernel-managed VA window, calls the real `libc::memset` to
/// fill it with a known byte, and reads byte 0 of that same block straight
/// back out through the guest's own load instruction — proving
/// `sceKernelMapFlexibleMemory` now allocates real, dereferenceable guest
/// memory (RT2 Task 5; it no longer returns the old fake-address stub) and
/// that `memset` actually wrote through it, all via the genuine VEH
/// trap-and-dispatch path (no test-only accessor into the arena).
///
/// Separately — using the *same* [`OrbisKernel`] instance passed to
/// `execute_linked` — asserts that `kernel.memory.is_mapped` reflects the
/// mapping `sceKernelMapFlexibleMemory` recorded, and that the mapped address
/// stays below the title libc limit rather than leaking the emulator's 16 TiB
/// image-arena address.
#[test]
fn mmap_then_memset_then_readback_proves_real_arena_memory_and_records_vmm_metadata() {
    const ENTRY_OFF: usize = 0x0;
    const SLOT_MAP_OFF: usize = 0x80;
    const SLOT_MEMSET_OFF: usize = 0x88;
    const SCRATCH_OFF: usize = 0x90;
    const MMAP_LEN: u64 = 0x40;
    const MMAP_PROT: u32 = 0x3; // R+W
    const MEMSET_VALUE: u32 = 0xCD;

    const SYSTEM_MANAGED_MIN: u64 = 0x0040_0000;
    const NATIVE_LIBC_LIMIT: u64 = 0x00FB_FFC0_0000;

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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libkernel", "libc"],
    );

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
    // recorded the mapping's metadata (record_mapping tags it "arena_mmap"),
    // and the recorded address must be valid console VA.
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
    assert!(
        mapped_addr >= SYSTEM_MANAGED_MIN && mapped_addr + MMAP_LEN <= NATIVE_LIBC_LIMIT,
        "mapped address {mapped_addr:#x} must stay in native-libc-compatible console VA space"
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
/// `raeen_runtime::arena`'s private constants).
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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libtest"],
    );

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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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

/// M1-E C1 acceptance: returning guest code may destroy every SysV
/// callee-saved GPR and must still recover the host context. A second run on
/// the same host thread proves recovery did not leave latent corruption.
#[test]
fn guest_return_recovers_host_context_through_trampoline() {
    const RETURN_VALUE: u64 = 0x1122_3344_5566_7788;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    let mut off = 0;
    for prefix in [
        [0x48, 0xBB], // rbx
        [0x48, 0xBD], // rbp
        [0x49, 0xBC], // r12
        [0x49, 0xBD], // r13
        [0x49, 0xBE], // r14
        [0x49, 0xBF], // r15
    ] {
        image[off..off + 2].copy_from_slice(&prefix);
        image[off + 2..off + 10].copy_from_slice(&0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
        off += 10;
    }
    image[off..off + 2].copy_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    image[off + 2..off + 10].copy_from_slice(&RETURN_VALUE.to_le_bytes());
    image[off + 10] = 0xC3; // ret

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };
    let kernel = OrbisKernel::new();

    for run in 1..=2 {
        assert_eq!(
            execute_linked(&linked, &hle, &kernel, 0, &[])
                .expect("guarded guest return must recover the host context"),
            RETURN_VALUE,
            "run {run} must preserve the guest retval while restoring all host registers"
        );
    }
}

// --- RT2c-b: TLS via fsbase (design doc §3/§5's acceptance tests) ---
//
// `raeen_runtime::tls` is a private module, so this black-box test file
// duplicates the same minimal FSGSBASE probe/read it uses internally (the
// same pattern already used elsewhere in this file for private arena
// constants like `HEAP_OFFSET`/`STACK_OFFSET`) -- purely to (a) gate these
// tests on FSGSBASE actually being available on the machine running them,
// since that's environment-dependent, and (b) observe the *host* thread's FS
// base for the "no leak"/"restored" assertions, which requires reading it
// from outside `execute_linked` entirely.

/// Duplicate of `raeen_runtime::tls::fsgsbase_available`'s CPUID probe.
fn fsgsbase_available() -> bool {
    // SAFETY: `__cpuid_count` is a safe fn on this target/rustc -- `CPUID`
    // is unconditionally available on x86-64; leaf 7/sub-leaf 0 is a
    // standard, always-queryable "Extended Features" leaf.
    let regs = core::arch::x86_64::__cpuid_count(7, 0);
    (regs.ebx & 1) != 0
}

/// Duplicate of `raeen_runtime::tls::read_fsbase` -- reads the *host*
/// thread's current FS base, used only to observe that `execute_linked`
/// restores it after a guest call.
///
/// # Safety
/// Caller must have confirmed `fsgsbase_available()` returns `true` first.
unsafe fn read_host_fsbase() -> u64 {
    let value: u64;
    // SAFETY: per this function's contract, the caller has confirmed
    // FSGSBASE is available. Same `RDFSBASE RAX` encoding
    // (`F3 48 0F AE C0`) as `raeen_runtime::tls::read_fsbase`.
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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

/// M1-E I3 acceptance: guest TLS keeps working **across a preemption**.
///
/// `tls.rs`'s `fsbase_does_not_survive_preemption_on_windows` pins the platform
/// reality: Windows discards a user-set FS base at the first context switch (a
/// bare timer-interrupt preemption suffices — no syscall). Without the
/// `veh_callback` re-arm, this guest's `mov rax, fs:[0]` after a long spin reads
/// a near-null address, the VEH sees a genuine wild fault, and the run comes
/// back `Err(Faulted)` — i.e. every real (longer-than-a-quantum) title that
/// touches TLS or its `fs:0x28` stack canary would "crash" within ~15ms.
///
/// The guest here spins in pure user mode (no syscall — so this exercises real
/// timer-interrupt preemption, not a kernel transition) while host spinners
/// saturate every core to guarantee it is actually descheduled, then reads
/// `fs:[0]`. Passing proves the fault-driven re-arm restored the FS base and
/// transparently retried the faulting instruction.
#[test]
fn guest_tls_survives_preemption_via_fsbase_rearm() {
    if !fsgsbase_available() {
        println!("FSGSBASE not available on this CPU; skipping fsbase re-arm test");
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const HEAP_OFFSET: u64 = 0x4000_0000;
    // Must span SEVERAL scheduler quanta, not one — see the identical constant
    // in `genuine_wild_fault_after_preemption_recovers_instead_of_looping_the_veh`.
    // At the original 1e8 (~30ms, i.e. about one quantum) this test passed on
    // roughly half of runs *without the re-arm ever firing*: the guest simply
    // was not preempted, so its FS base was never cleared and `fs:[0]` resolved
    // directly. It looked like an acceptance gate while proving nothing. ~2e9
    // (~600ms) makes preemption certain, and the `rearms_after > rearms_before`
    // assertion below fails loudly if it ever stops happening.
    const SPIN_COUNT: u64 = 2_000_000_000;

    let hle = HleRegistry::new();
    let mut code: Vec<u8> = Vec::new();
    // mov rcx, SPIN_COUNT
    code.extend_from_slice(&[0x48, 0xB9]);
    code.extend_from_slice(&SPIN_COUNT.to_le_bytes());
    // spin: dec rcx ; jnz spin   (rel8 = -5: back over jnz(2) + dec(3))
    code.extend_from_slice(&[0x48, 0xFF, 0xC9]);
    code.extend_from_slice(&[0x75, 0xFB]);
    // mov rax, fs:[0]   — the TCB self-pointer read that traps if the base was
    // discarded and never re-armed.
    code.extend_from_slice(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00]);
    // ret
    code.push(0xC3);

    let mut image = vec![0u8; 0x100];
    image[ENTRY_OFF..ENTRY_OFF + code.len()].copy_from_slice(&code);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    // Saturate every core so the guest thread is genuinely preempted rather
    // than running its spin to completion uninterrupted on an idle machine.
    // `Spinners` stops+joins on drop, so an assertion panic below can never
    // leak the busy-loops (which would peg every core for the rest of the test
    // binary's run and even lock its output .exe against relinking).
    let spinners = Spinners::contend_one_core();

    let kernel = OrbisKernel::new();
    let rearms_before = fsbase_rearm_count();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]);
    let rearms_after = fsbase_rearm_count();

    drop(spinners);

    let expected_tcb = GUEST_ARENA_BASE + HEAP_OFFSET;
    let rax = result.expect(
        "guest must not fault after preemption: the VEH must re-arm the FS base \
         Windows discarded and retry the faulting fs:-relative access",
    );
    assert_eq!(
        rax, expected_tcb,
        "after a preemption discarded the FS base, the re-armed `mov rax, fs:[0]` \
         must still read the installed TCB self-pointer"
    );
    // Guard against passing for the wrong reason: had no preemption landed
    // during the spin, the base would still match, the re-arm arm would never
    // run, and `fs:[0]` would read the TCB without exercising the mechanism.
    assert!(
        rearms_after > rearms_before,
        "the spin must actually have been preempted (and the FS base re-armed) — \
         otherwise this test passes without exercising the re-arm at all"
    );
}

/// M1-E I3, termination guard: a **genuine** wild fault that happens *after* a
/// preemption discarded the FS base must still recover as `Faulted` — not spin
/// the VEH forever.
///
/// This pins the load-bearing platform property the re-arm's termination
/// argument rests on. `veh_callback`'s re-arm arm fires on *any* out-of-region
/// fault whose current FS base differs from the guest TCB, so a post-preemption
/// wild access takes this path:
///
/// 1. trap 1: base was cleared by the preemption, so `base != guest_fsbase` →
///    re-arm + `EXCEPTION_CONTINUE_EXECUTION` → the faulting instruction retries;
/// 2. trap 2: it faults again, but now `base == guest_fsbase`, so the arm is
///    skipped and the genuine-fault (RT1a) recovery runs.
///
/// Step 2 only terminates if a `WRFSBASE` issued **inside the handler** survives
/// the `EXCEPTION_CONTINUE_EXECUTION` return. If it does not, trap 2 re-arms
/// again, and again, forever: an **infinite VEH loop — a hang, not a crash**.
/// The existing `genuine_wild_fault_recovers_as_faulted_then_process_keeps_running`
/// does NOT cover this: its guest faults on its very first instruction, before
/// any preemption, so the base still matches and the re-arm arm never runs.
///
/// (`guest_tls_survives_preemption_via_fsbase_rearm` proves the same platform
/// property from the success side — the retried `fs:[0]` reads the TCB. This
/// test proves it from the failure side, where a regression hangs the suite
/// rather than failing an assert, so it is worth pinning explicitly.)
#[test]
fn genuine_wild_fault_after_preemption_recovers_instead_of_looping_the_veh() {
    if !fsgsbase_available() {
        println!("FSGSBASE not available on this CPU; skipping fsbase re-arm termination test");
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    // Must span SEVERAL scheduler quanta, not one. A Windows foreground quantum
    // is ~20-46ms; a fused `dec`/`jnz` retires ~1/cycle, so 1e8 iterations is
    // only ~30ms — comparable to a single quantum, which made preemption (and
    // therefore the whole path under test) a coin flip depending on where in its
    // quantum the guest thread started. Measured: at 1e8 the re-arm fired on
    // roughly half of runs. ~2e9 is ~600ms = many quanta, so preemption is
    // certain rather than lucky. The `rearms_after > rearms_before` assertion
    // below is what turns a regression here into a failure instead of a silent
    // vacuous pass.
    const SPIN_COUNT: u64 = 2_000_000_000;

    let hle = HleRegistry::new();
    let mut code: Vec<u8> = Vec::new();
    // mov rcx, SPIN_COUNT
    code.extend_from_slice(&[0x48, 0xB9]);
    code.extend_from_slice(&SPIN_COUNT.to_le_bytes());
    // spin: dec rcx ; jnz spin
    code.extend_from_slice(&[0x48, 0xFF, 0xC9]);
    code.extend_from_slice(&[0x75, 0xFB]);
    // The wild dereference of address 0 — a genuine fault, NOT an fs: access,
    // reached only after the spin above has been preempted.
    let fault_off = code.len();
    code.extend_from_slice(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00]);
    // ret
    code.push(0xC3);

    let mut image = vec![0u8; 0x100];
    image[ENTRY_OFF..ENTRY_OFF + code.len()].copy_from_slice(&code);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    // Saturate every core so the guest is genuinely descheduled during the spin.
    let spinners = Spinners::contend_one_core();

    let kernel = OrbisKernel::new();
    let rearms_before = fsbase_rearm_count();
    let result = execute_linked(&linked, &hle, &kernel, ENTRY_OFF as u64, &[]);
    let rearms_after = fsbase_rearm_count();

    drop(spinners);

    // Reaching this line at all is most of the point: a non-terminating re-arm
    // never returns from `execute_linked`.
    let err = result.expect_err("a wild dereference of address 0 must be reported as a fault");
    assert_eq!(
        err,
        RuntimeError::Faulted {
            addr: GUEST_ARENA_BASE + fault_off as u64,
            access: 0,
            kind: raeen_runtime::FaultKind::Read,
        },
        "the post-preemption genuine fault must be recovered as `Faulted` at the \
         faulting instruction, i.e. the FS-base re-arm must fall through to the \
         RT1a path on the retry rather than re-arming forever"
    );
    // Without this, the test would pass vacuously on a run where no preemption
    // landed: the base would still match, the re-arm arm would be skipped, and
    // the wild fault would reach the RT1a path directly — proving nothing about
    // whether the re-arm terminates.
    assert!(
        rearms_after > rearms_before,
        "the spin must actually have been preempted (clearing the FS base) so that \
         the wild fault below it goes through the re-arm-then-retry path — otherwise \
         this test does not exercise the loop it is meant to guard against"
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
// `2026-07-13-raeen-crt0-process-env-design.md` §6/§7) ---
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

/// The **Orbis (PS4/PS5) `_start` ABI**: the entry is called like a function,
/// `_start(EntryParams *params /* rdi */, void (*exit_fn)(void) /* rsi */)`,
/// where `params` points at the `argc, argv[], ...` block — it does NOT read
/// `argc` off the stack the way a Linux `_start` does.
///
/// This is not a guess: a real retail PS5 title's entry begins
/// `push rbp; mov rbp,rsp; push r15; push r14; push rbx; push rax;`
/// `mov r14d,[rdi]; mov rbx,rsi; lea r15,[rdi+8]` — i.e. argc from `[rdi]`,
/// argv at `rdi+8`, exit_fn in `rsi`. With `rdi` left at 0 (as a bare `jmp`
/// leaves it) that is a null dereference, and the title died ~10 bytes into
/// its own entry before this was fixed.
///
/// Mirrors the `[rsp]` test below, but reads `argc` through `rdi`:
/// `mov eax,[rdi]; jmp rax` faults with `Rip == argc == 1`.
#[test]
fn start_stub_observes_argc_via_rdi_per_the_orbis_entry_abi() {
    const ENTRY_OFF: usize = 0x0;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    // mov eax, [rdi]   (8B 07)   -- argc, exactly like a real title's `mov r14d,[rdi]`
    image[ENTRY_OFF..ENTRY_OFF + 2].copy_from_slice(&[0x8B, 0x07]);
    // jmp rax          (FF E0)
    image[ENTRY_OFF + 2..ENTRY_OFF + 4].copy_from_slice(&[0xFF, 0xE0]);

    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr, .. } => assert_eq!(
            addr, 1,
            "argc read through rdi must equal 1 — the Orbis entry ABI passes the \
             params block in rdi, not (only) on the stack"
        ),
        other => panic!("expected Err(Faulted {{ .. }}), got {other:?}"),
    }
}

/// The second Orbis `_start` argument is a process-exit callback. Retail crt0
/// preserves RSI and calls through it on fatal startup paths; passing zero
/// turns a clean termination into an execute-at-null fault.
#[test]
fn process_entry_receives_a_working_exit_callback_in_rsi() {
    const EXIT_CODE: u32 = 0x2a;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    // mov edi, EXIT_CODE; jmp rsi
    image[..7].copy_from_slice(&[
        0xBF,
        EXIT_CODE as u8,
        (EXIT_CODE >> 8) as u8,
        (EXIT_CODE >> 16) as u8,
        (EXIT_CODE >> 24) as u8,
        0xFF,
        0xE6,
    ]);
    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("jumping through the supplied exit callback must terminate cleanly");
    assert_eq!(outcome, RunOutcome::Exited(u64::from(EXIT_CODE)));
}

/// Orbis enters `_start` with the alignment of an ordinary called SysV
/// function: `rsp % 16 == 8`. Real retail code relies on this for aligned
/// XMM spills after `push rbp`; entering at 0 mod 16 shifts every frame by
/// eight and turns `vmovaps` into a #GP (reported by Windows as access -1).
#[test]
fn process_entry_has_orbis_called_function_stack_alignment() {
    const ENTRY_OFF: usize = 0;

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    // mov rax, rsp; and eax, 0xf; jmp rax
    image[..8].copy_from_slice(&[0x48, 0x89, 0xE0, 0x83, 0xE0, 0x0F, 0xFF, 0xE0]);
    let linked = LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect_err("jumping to rsp & 0xf must fault and expose the alignment");
    match err {
        RuntimeError::Faulted { addr, .. } => assert_eq!(addr, 8),
        other => panic!("expected alignment observation fault, got {other:?}"),
    }
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr, .. } => {
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr, .. } => {
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: Vec::<HleTrampoline>::new(),
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    };

    let kernel = OrbisKernel::new();
    let err = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[]).unwrap_err();
    match err {
        RuntimeError::Faulted { addr, .. } => {
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libc"],
    );

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

/// Multi-module static TLS acceptance: a DEPENDENCY's `.tdata` must be
/// materialized at ITS assigned slot below the thread pointer, not folded onto
/// the main module's block.
///
/// The guest reads `fs:[0x8 - 0x60]` — the initial-exec address of a variable
/// at template offset 0x8 in a module assigned `tp_offset = 0x60` — and exits
/// with it. Before the process-wide layout existed, only the main module's
/// template was ever copied, so this read returned zero: exactly how libc.prx's
/// 0x188 bytes of initialized TLS (errno, locale, strtok state) silently read
/// back as garbage on the measured retail title.
#[test]
fn dependency_tdata_is_materialized_at_its_static_tls_slot() {
    if !fsgsbase_available() {
        println!(
            "FSGSBASE not available on this CPU; skipping dependency_tdata_is_materialized_at_its_static_tls_slot"
        );
        return;
    }

    const ENTRY_OFF: usize = 0x0;
    const SLOT_EXIT_OFF: usize = 0x40;
    const SLOT_TLS_OFF: usize = 0x48;
    const DEP_TLS_VALUE: u64 = 0xDE9_0DA7A;
    const DEP_VAR_OFF: u64 = 0x8;
    const DEP_TP_OFFSET: u64 = 0x60;

    let mut image = vec![0u8; 0x100];
    let mut off = ENTRY_OFF;
    // mov rax, [rip+disp32] -> SLOT_TLS_OFF (holds the fs-relative offset)
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
    // The initial-exec offset the linker would have computed against the
    // dependency's assignment: var - tp_offset. Written as plain data (no
    // relocation) — this test exercises the runtime's block building, not the
    // linker (linker.rs pins the TPOFF64 arithmetic separately).
    let fs_offset = DEP_VAR_OFF.wrapping_sub(DEP_TP_OFFSET);
    image[SLOT_TLS_OFF..SLOT_TLS_OFF + 8].copy_from_slice(&fs_offset.to_le_bytes());

    // The main module's own template: 8 bytes of .tdata that must ALSO land
    // correctly (at tp-0x20), proving the two coexist.
    let main_tdata = vec![0x11u8; 0x8];
    let module = SprxModule {
        name: "tlsMainModule".to_string(),
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
            data: main_tdata,
            mem_size: 0x18,
            align: 0x8,
        }),
        procparam: None,
        unwind: None,
    };

    let dynlib = bind_import_providers(
        DynlibData {
            symbols: vec![DynSymbol {
                nid: nid_of("exit"),
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: SLOT_EXIT_OFF as u64,
                info: R_X86_64_JUMP_SLOT, // r_sym = 0 -> the exit import
                addend: 0,
            }],
            ..Default::default()
        },
        &["libc"],
    );

    let hle = HleRegistry::new();
    let db = NidDatabase::from_hle_names(hle.registered_names());
    let registry = ModuleRegistry::new(db);
    let mut linked = link_module(&module, &dynlib, &registry, &hle, GUEST_ARENA_BASE)
        .expect("the main module links");

    // The process layout `load_process` would have computed: main at tp-0x20,
    // the dependency below it at tp-0x60, its `.tdata` holding DEP_TLS_VALUE
    // at template offset DEP_VAR_OFF.
    let mut dep_tdata = vec![0u8; 0x10];
    dep_tdata[DEP_VAR_OFF as usize..DEP_VAR_OFF as usize + 8]
        .copy_from_slice(&DEP_TLS_VALUE.to_le_bytes());
    linked.tls_layout.push(raeen_firmware::StaticTlsModule {
        name: "libdep.prx".to_string(),
        module_id: 2,
        tp_offset: DEP_TP_OFFSET,
        template: TlsTemplate {
            vaddr: 0,
            data: dep_tdata,
            mem_size: 0x18,
            align: 0x8,
        },
    });

    let kernel = OrbisKernel::new();
    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("dependency TLS read must not fault");
    assert_eq!(
        outcome,
        RunOutcome::Exited(DEP_TLS_VALUE),
        "the dependency's .tdata must be readable at its own fs-relative slot"
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
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        }],
        entry: ENTRY_OFF as u64,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
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
        unwind: None,
    };
    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libc", "libc"],
    );

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
        unwind: None,
    };
    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libkernel", "libc"],
    );

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
        unwind: None,
    };
    let dynlib = bind_import_providers(
        DynlibData {
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
        },
        &["libc", "libc"],
    );

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

// ---------------------------------------------------------------------------
// Slice 1: module initialization runs exactly once.
//
// A retail crt0 `_start` walks the executable's own init array itself, so a
// process loader that ALSO calls the main initializer double-constructs the
// title's globals. Measured on ASTRO.BOT, a list-building constructor then
// formed a cyclic list its own later walk hung on forever. These fixtures
// prove: the process loader runs each DEPENDENCY initializer once and
// WITHHOLDS the main executable's (crt0 owns it), while a crt0-less direct
// execution (`execute_linked`) is loader-owned and runs everything.
//
// Observability: each synthetic initializer is a guest `call [slot]; ret` into
// a distinct HLE trampoline that bumps a host-side counter — the same
// trap-and-dispatch channel every other test here uses. A dedicated lock
// serializes just these tests so one test's counter reset never races
// another's read (the runtime's own `call_lock` serializes guest execution but
// not this bookkeeping).
// ---------------------------------------------------------------------------

/// Serializes the initializer-counter tests so they can share the host-side
/// call counters below without one test's reset racing another's assertion.
static INIT_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
static DEP_INIT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MAIN_INIT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// HLE function the synthetic DEPENDENCY initializer calls — bumps a host counter.
fn record_dep_init(_ctx: &HleContext, _args: &[u64]) -> u64 {
    DEP_INIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    0
}

/// HLE function the synthetic MAIN initializer calls — bumps a host counter.
fn record_main_init(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MAIN_INIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    0
}

// Image layout shared by the initializer fixtures (image is 0x100 bytes).
const INIT_START_OFF: usize = 0x00; // `_start` (process-mode entry)
const INIT_LINKED_ENTRY_OFF: usize = 0x20; // trivial `mov eax,0xAB; ret` (function mode)
const INIT_DEP_FN_OFF: usize = 0x40; // dependency initializer: call [DEP_SLOT]; ret
const INIT_MAIN_FN_OFF: usize = 0x50; // main initializer: call [MAIN_SLOT]; ret
const INIT_DEP_SLOT: usize = 0x60; // -> HLE_TRAMPOLINE_BASE + 0  (depInit)
const INIT_MAIN_SLOT: usize = 0x68; // -> HLE_TRAMPOLINE_BASE + 8  (mainInit)
const INIT_EXIT_SLOT: usize = 0x70; // -> HLE_TRAMPOLINE_BASE + 16 (libc::exit)
const INIT_LINKED_SENTINEL: u32 = 0xAB;

/// `call qword ptr [rip+disp32]` (`FF 15 <disp32>`, 6 bytes) at `at`, targeting
/// the 8-byte slot at `slot_off`. Returns the offset just past the instruction.
fn write_call_indirect(buf: &mut [u8], at: usize, slot_off: usize) -> usize {
    let rip_after = at as i64 + 6;
    let disp32 = (slot_off as i64 - rip_after) as i32;
    buf[at] = 0xFF;
    buf[at + 1] = 0x15;
    buf[at + 2..at + 6].copy_from_slice(&disp32.to_le_bytes());
    at + 6
}

/// `call rel32` (`E8 <rel32>`, 5 bytes) at `at`, targeting `target`. Returns the
/// offset just past the instruction.
fn write_call_rel(buf: &mut [u8], at: usize, target: usize) -> usize {
    let rip_after = at as i64 + 5;
    let rel32 = (target as i64 - rip_after) as i32;
    buf[at] = 0xE8;
    buf[at + 1..at + 5].copy_from_slice(&rel32.to_le_bytes());
    at + 5
}

/// `mov edi, imm32` (`BF <imm32>`, 5 bytes) at `at`. Returns the next offset.
fn write_mov_edi(buf: &mut [u8], at: usize, imm: u32) -> usize {
    buf[at] = 0xBF;
    buf[at + 1..at + 5].copy_from_slice(&imm.to_le_bytes());
    at + 5
}

/// Build the shared initializer fixture image. `start_runs_main` chooses the
/// `_start` shape: `true` emulates a crt0 that re-runs the executable's own
/// initializer (`call INIT_MAIN_FN_OFF`) before exiting; `false` exits without
/// touching it (proving the loader's own decision in isolation).
fn build_init_image(start_runs_main: bool, exit_code: u32) -> Vec<u8> {
    let mut img = vec![0u8; 0x100];

    // Dependency initializer @0x40: call [DEP_SLOT]; ret
    let past = write_call_indirect(&mut img, INIT_DEP_FN_OFF, INIT_DEP_SLOT);
    img[past] = 0xC3;
    // Main initializer @0x50: call [MAIN_SLOT]; ret
    let past = write_call_indirect(&mut img, INIT_MAIN_FN_OFF, INIT_MAIN_SLOT);
    img[past] = 0xC3;
    // Function-mode entry @0x20: mov eax, 0xAB; ret
    img[INIT_LINKED_ENTRY_OFF] = 0xB8;
    img[INIT_LINKED_ENTRY_OFF + 1..INIT_LINKED_ENTRY_OFF + 5]
        .copy_from_slice(&INIT_LINKED_SENTINEL.to_le_bytes());
    img[INIT_LINKED_ENTRY_OFF + 5] = 0xC3;

    // Trampoline slots (index i -> HLE_TRAMPOLINE_BASE + i*8, per trampoline::resolve).
    img[INIT_DEP_SLOT..INIT_DEP_SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    img[INIT_MAIN_SLOT..INIT_MAIN_SLOT + 8]
        .copy_from_slice(&(HLE_TRAMPOLINE_BASE + 8).to_le_bytes());
    img[INIT_EXIT_SLOT..INIT_EXIT_SLOT + 8]
        .copy_from_slice(&(HLE_TRAMPOLINE_BASE + 16).to_le_bytes());

    // `_start` @0x00.
    let mut at = INIT_START_OFF;
    if start_runs_main {
        at = write_call_rel(&mut img, at, INIT_MAIN_FN_OFF);
    }
    at = write_mov_edi(&mut img, at, exit_code);
    let _ = write_call_indirect(&mut img, at, INIT_EXIT_SLOT);

    img
}

/// The three trampolines the fixture image addresses, in slot order.
fn init_trampolines() -> Vec<HleTrampoline> {
    vec![
        HleTrampoline {
            library: "libtest".to_string(),
            function: "depInit".to_string(),
            addr: HLE_TRAMPOLINE_BASE,
        },
        HleTrampoline {
            library: "libtest".to_string(),
            function: "mainInit".to_string(),
            addr: HLE_TRAMPOLINE_BASE + 8,
        },
        HleTrampoline {
            library: "libc".to_string(),
            function: "exit".to_string(),
            addr: HLE_TRAMPOLINE_BASE + 16,
        },
    ]
}

fn register_init_counters(hle: &HleRegistry) {
    hle.register("libtest", "depInit", record_dep_init);
    hle.register("libtest", "mainInit", record_main_init);
}

fn init_linked_module(image: Vec<u8>, entry: u64, module_inits: Vec<ModuleInit>) -> LinkedModule {
    LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits,
        hle_trampolines: init_trampolines(),
        entry,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    }
}

/// A dependency initializer scheduled before the main executable's, as
/// `load_process` builds it.
fn process_module_inits() -> Vec<ModuleInit> {
    vec![
        ModuleInit {
            name: "libdep.prx".to_string(),
            image_offset: INIT_DEP_FN_OFF as u64,
            role: ModuleInitRole::Dependency,
        },
        ModuleInit {
            name: "eboot.bin".to_string(),
            image_offset: INIT_MAIN_FN_OFF as u64,
            role: ModuleInitRole::Main,
        },
    ]
}

/// The process loader runs each dependency initializer once and must NOT call
/// the main executable's own initializer — crt0 owns it. Here `_start` never
/// runs the main initializer, so the main counter isolates the loader's choice:
/// it must be zero. (Against the pre-Slice-1 loop it was one — the loop called
/// the last `module_inits` entry unconditionally.)
#[test]
fn execute_process_does_not_call_the_main_initializer_itself() {
    let _serialize = INIT_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    DEP_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
    MAIN_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);

    let hle = HleRegistry::new();
    register_init_counters(&hle);
    let linked = init_linked_module(
        build_init_image(false, 7),
        INIT_START_OFF as u64,
        process_module_inits(),
    );
    let kernel = OrbisKernel::new();

    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("process must not fault");
    assert_eq!(outcome, RunOutcome::Exited(7));
    assert_eq!(
        DEP_INIT_CALLS.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the loader runs the dependency initializer exactly once"
    );
    assert_eq!(
        MAIN_INIT_CALLS.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the process loader must NOT call the main executable's initializer — crt0 owns it"
    );
}

/// With a crt0 `_start` that DOES run the executable's initializer (the retail
/// shape), the main initializer must fire EXACTLY once. The loader withholds
/// its own call, so crt0's single run is the only one — no double-init, which
/// is what built ASTRO.BOT's cyclic constructor list. (Against the pre-Slice-1
/// loop this counter reached two: loader + crt0.)
#[test]
fn execute_process_runs_the_main_initializer_exactly_once_via_crt0() {
    let _serialize = INIT_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    DEP_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
    MAIN_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);

    let hle = HleRegistry::new();
    register_init_counters(&hle);
    let linked = init_linked_module(
        build_init_image(true, 9),
        INIT_START_OFF as u64,
        process_module_inits(),
    );
    let kernel = OrbisKernel::new();

    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("process must not fault");
    assert_eq!(outcome, RunOutcome::Exited(9));
    assert_eq!(
        DEP_INIT_CALLS.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the dependency initializer still runs once"
    );
    assert_eq!(
        MAIN_INIT_CALLS.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the main initializer must run EXACTLY once: crt0 runs it and the loader must not also"
    );
}

/// Direct function/module execution enters no crt0, so it is loader-owned: with
/// nothing else to run the main initializer, `execute_linked` runs it (and
/// every other initializer) before entering the requested function. Proves the
/// `LoaderOwnsMainInit` branch and that the requested entry still runs.
#[test]
fn execute_linked_runs_loader_owned_initializers_including_main() {
    let _serialize = INIT_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    DEP_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
    MAIN_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);

    let hle = HleRegistry::new();
    register_init_counters(&hle);
    let linked = init_linked_module(
        build_init_image(false, 0),
        INIT_LINKED_ENTRY_OFF as u64,
        vec![ModuleInit {
            name: "eboot.bin".to_string(),
            image_offset: INIT_MAIN_FN_OFF as u64,
            role: ModuleInitRole::Main,
        }],
    );
    let kernel = OrbisKernel::new();

    let result = execute_linked(&linked, &hle, &kernel, INIT_LINKED_ENTRY_OFF as u64, &[])
        .expect("direct execution must not fault");
    assert_eq!(
        result, INIT_LINKED_SENTINEL as u64,
        "the requested entry function ran and returned its sentinel"
    );
    assert_eq!(
        MAIN_INIT_CALLS.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "direct execution is loader-owned: with no crt0, the loader runs the main initializer"
    );
}

/// The deterministic diagnostic stream records each initializer transition —
/// the dependency's run and the main executable's deferral — with module name,
/// role, and a stable, monotonic sequence number (Slice 1 point 5).
#[test]
fn execute_process_records_initializer_transitions_in_diagnostic_mode() {
    use raeen_core::diagnostics::{DiagnosticKind, DiagnosticRecorder};

    let _serialize = INIT_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    DEP_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
    MAIN_INIT_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);

    let hle = HleRegistry::new();
    register_init_counters(&hle);
    let linked = init_linked_module(
        build_init_image(false, 3),
        INIT_START_OFF as u64,
        process_module_inits(),
    );
    let mut kernel = OrbisKernel::new();
    kernel.diagnostics = std::sync::Arc::new(DiagnosticRecorder::new(true, 1024));

    let outcome = execute_process(&linked, &hle, &kernel, &["/app/eboot.bin"], &[])
        .expect("process must not fault");
    assert_eq!(outcome, RunOutcome::Exited(3));

    let events: Vec<_> = kernel
        .diagnostics
        .snapshot()
        .into_iter()
        .filter(|event| event.kind == DiagnosticKind::ModuleInit)
        .collect();
    assert_eq!(
        events.len(),
        2,
        "one transition per scheduled initializer: the dependency run and the main deferral"
    );

    assert_eq!(events[0].subject, "libdep.prx");
    assert_eq!(events[0].guest_thread, 1);
    assert!(
        events[0].detail.contains("role=dependency") && events[0].detail.contains("run"),
        "dependency initializer recorded as run, got {:?}",
        events[0].detail
    );

    assert_eq!(events[1].subject, "eboot.bin");
    assert!(
        events[1].detail.contains("role=main") && events[1].detail.contains("deferred-to-crt0"),
        "main initializer recorded as deferred to crt0, got {:?}",
        events[1].detail
    );

    assert!(
        events[1].sequence > events[0].sequence,
        "diagnostic sequence numbers are stable and monotonic"
    );
}

// ---------------------------------------------------------------------------
// Checklist item 7: SYNCHRONOUS guest callbacks — an HLE handler calls back
// INTO guest code mid-call via `GuestCallScheduler::call_guest` and receives
// its RAX, on the current guest thread. Acceptance tests below pin: the
// returned value reaching the handler (and the handler's own eventual return
// value reaching the guest), depth-2 nesting (HLE → callback → HLE →
// callback), fault propagation, `request_exit` unwind composition, the
// direct-gateway refusal, and the first real consumer (`qsort`).
// ---------------------------------------------------------------------------

/// `lea <reg>, [rip+disp32]` targeting `target` (an offset in the same flat
/// image). ModRM reg codes: rax=0 rcx=1 rdx=2 rbx=3 rsi=6 rdi=7.
fn emit_lea_rip(image: &mut [u8], off: &mut usize, modrm_reg: u8, target: usize) {
    let after = *off as i64 + 7;
    let disp = (target as i64 - after) as i32;
    image[*off..*off + 3].copy_from_slice(&[0x48, 0x8D, 0x05 | (modrm_reg << 3)]);
    image[*off + 3..*off + 7].copy_from_slice(&disp.to_le_bytes());
    *off += 7;
}

/// A [`LinkedModule`] over a hand-assembled image whose import slots were
/// already patched with `HLE_TRAMPOLINE_BASE + i*8`; trampoline `i` is the
/// `i`-th `(library, function)` pair.
fn callback_fixture_module(image: Vec<u8>, imports: &[(&str, &str)]) -> LinkedModule {
    LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: imports
            .iter()
            .enumerate()
            .map(|(i, (library, function))| HleTrampoline {
                library: (*library).to_string(),
                function: (*function).to_string(),
                addr: HLE_TRAMPOLINE_BASE + i as u64 * 8,
            })
            .collect(),
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    }
}

/// HLE handler: synchronously call the guest function in arg0 with arg1,
/// then return `callback_rax + 0x1000` — proving BOTH directions of the
/// value flow (callback result into the handler; the handler's eventual
/// result back to the interrupted guest call site).
fn hle_call_guest_and_add(ctx: &HleContext, args: &[u64]) -> u64 {
    match ctx
        .guest_calls
        .call_guest(args[0], [args[1], 0, 0, 0, 0, 0])
    {
        Ok(value) => value.wrapping_add(0x1000),
        Err(_) => 0xDEAD,
    }
}

/// A synchronous guest callback's RAX reaches the HLE handler mid-call, and
/// the handler's own return value still reaches the interrupted guest call.
#[test]
fn call_guest_returns_callback_rax_to_the_hle_handler() {
    const CB: usize = 0x20;
    const SLOT: usize = 0x40;

    let hle = HleRegistry::new();
    hle.register("libtest", "sceTestCallGuestAdd", hle_call_guest_and_add);

    let mut image = vec![0u8; 0x60];
    let mut off = 0usize;
    emit_lea_rip(&mut image, &mut off, 7, CB); // rdi = &callback
    image[off..off + 5].copy_from_slice(&[0xBE, 0x2A, 0x00, 0x00, 0x00]); // mov esi, 0x2A
    off += 5;
    emit_indirect_call(&mut image, &mut off, SLOT);
    image[off] = 0xC3; // ret
    // callback: lea rax, [rdi+1]; ret  — returns its argument + 1.
    image[CB..CB + 5].copy_from_slice(&[0x48, 0x8D, 0x47, 0x01, 0xC3]);
    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = callback_fixture_module(image, &[("libtest", "sceTestCallGuestAdd")]);
    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("synchronous callback round-trip must succeed");
    assert_eq!(
        result,
        0x2A + 1 + 0x1000,
        "callback rax (arg+1) must reach the handler, and the handler's result the guest"
    );
}

/// Depth-2 nesting handlers: outer calls guest cb1 (which calls the inner
/// import, which calls guest cb2). Each level adds a distinct constant so the
/// final value proves every hop ran exactly once, in order.
fn hle_nest_outer(ctx: &HleContext, args: &[u64]) -> u64 {
    match ctx
        .guest_calls
        .call_guest(args[0], [args[1], args[2], 0, 0, 0, 0])
    {
        Ok(value) => value + 0x100,
        Err(_) => 0xDEAD,
    }
}

fn hle_nest_inner(ctx: &HleContext, args: &[u64]) -> u64 {
    match ctx
        .guest_calls
        .call_guest(args[0], [args[1], 0, 0, 0, 0, 0])
    {
        Ok(value) => value + 0x10,
        Err(_) => 0xDEAD,
    }
}

/// Documented supported nesting depth: 2 (HLE → guest callback → HLE → guest
/// callback). Chain: entry → OUTER(cb1, cb2, 5) → cb1 → INNER(cb2, 5) → cb2.
/// cb2(5)=8; INNER=0x18; cb1=0x19; OUTER=0x119.
#[test]
fn nested_call_guest_composes_to_depth_two() {
    const CB1: usize = 0x30;
    const CB2: usize = 0x50;
    const OUTER_SLOT: usize = 0x60;
    const INNER_SLOT: usize = 0x68;

    let hle = HleRegistry::new();
    hle.register("libtest", "sceTestNestOuter", hle_nest_outer);
    hle.register("libtest", "sceTestNestInner", hle_nest_inner);

    let mut image = vec![0u8; 0x80];
    let mut off = 0usize;
    emit_lea_rip(&mut image, &mut off, 7, CB1); // rdi = &cb1
    emit_lea_rip(&mut image, &mut off, 6, CB2); // rsi = &cb2
    image[off..off + 5].copy_from_slice(&[0xBA, 0x05, 0x00, 0x00, 0x00]); // mov edx, 5
    off += 5;
    emit_indirect_call(&mut image, &mut off, OUTER_SLOT);
    image[off] = 0xC3; // ret
    // cb1 (rdi = &cb2, rsi = 5): keep SysV 16-byte call alignment, forward
    // its own arguments to the INNER import, add 1 to its result.
    let mut cb1 = CB1;
    image[cb1..cb1 + 4].copy_from_slice(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8
    cb1 += 4;
    emit_indirect_call(&mut image, &mut cb1, INNER_SLOT);
    image[cb1..cb1 + 4].copy_from_slice(&[0x48, 0x83, 0xC4, 0x08]); // add rsp, 8
    cb1 += 4;
    image[cb1..cb1 + 4].copy_from_slice(&[0x48, 0x83, 0xC0, 0x01]); // add rax, 1
    cb1 += 4;
    image[cb1] = 0xC3; // ret
    assert!(cb1 < CB2, "cb1 must not overlap cb2");
    // cb2 (rdi = 5): lea rax, [rdi+3]; ret.
    image[CB2..CB2 + 5].copy_from_slice(&[0x48, 0x8D, 0x47, 0x03, 0xC3]);
    image[OUTER_SLOT..OUTER_SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    image[INNER_SLOT..INNER_SLOT + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + 8).to_le_bytes());

    let linked = callback_fixture_module(
        image,
        &[
            ("libtest", "sceTestNestOuter"),
            ("libtest", "sceTestNestInner"),
        ],
    );
    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("depth-2 nested callbacks must succeed");
    assert_eq!(
        result, 0x119,
        "every nesting level must run exactly once, in order"
    );
}

static FAULTING_CB_HANDLER_RESUMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn hle_call_guest_fault_probe(ctx: &HleContext, args: &[u64]) -> u64 {
    let result = ctx.guest_calls.call_guest(args[0], [0; 6]);
    // A faulting callback longjmps to the run's recovery point: this store
    // (and the handler's return) must never execute.
    FAULTING_CB_HANDLER_RESUMED.store(true, std::sync::atomic::Ordering::SeqCst);
    result.unwrap_or(0xDEAD)
}

/// A guest callback that faults must surface as `Err(Faulted)` from the run
/// — never as corruption, and never as a successful return to the HLE
/// handler that invoked it.
#[test]
fn faulting_guest_callback_propagates_as_a_fault_not_a_return() {
    const CB: usize = 0x20;
    const SLOT: usize = 0x40;
    const POISON: u32 = 0xBAD;

    let hle = HleRegistry::new();
    hle.register(
        "libtest",
        "sceTestCallGuestFault",
        hle_call_guest_fault_probe,
    );

    let mut image = vec![0u8; 0x60];
    let mut off = 0usize;
    emit_lea_rip(&mut image, &mut off, 7, CB); // rdi = &callback
    emit_indirect_call(&mut image, &mut off, SLOT);
    image[off] = 0xB8; // mov eax, POISON (must never execute)
    image[off + 1..off + 5].copy_from_slice(&POISON.to_le_bytes());
    image[off + 5] = 0xC3;
    // callback: mov rax, [0]; ret — a null read.
    image[CB..CB + 8].copy_from_slice(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00]);
    image[CB + 8] = 0xC3;
    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = callback_fixture_module(image, &[("libtest", "sceTestCallGuestFault")]);
    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[]);
    match result {
        Err(RuntimeError::Faulted { access, .. }) => {
            assert_eq!(access, 0, "the callback's null read is the reported access");
        }
        other => panic!("a faulting callback must recover as Err(Faulted), got {other:?}"),
    }
    assert!(
        !FAULTING_CB_HANDLER_RESUMED.load(std::sync::atomic::Ordering::SeqCst),
        "the HLE handler must never resume after its callback faulted"
    );
}

static EXITING_CB_HANDLER_RESUMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn hle_call_guest_exit_probe(ctx: &HleContext, args: &[u64]) -> u64 {
    let result = ctx.guest_calls.call_guest(args[0], [0; 6]);
    // The callback triggers __stack_chk_fail's request_exit unwind: this
    // store (and the handler's return) must never execute.
    EXITING_CB_HANDLER_RESUMED.store(true, std::sync::atomic::Ordering::SeqCst);
    result.unwrap_or(0xDEAD)
}

/// A callback that triggers the `request_exit` unwind (`__stack_chk_fail`)
/// must unwind the whole guest call cleanly — the interrupted HLE handler
/// never resumes, and the run ends with the fatal exit code, exactly like a
/// canary smash outside a callback.
#[test]
fn callback_that_requests_exit_unwinds_past_the_interrupted_hle_handler() {
    const CB: usize = 0x20;
    const PROBE_SLOT: usize = 0x40;
    const CHK_SLOT: usize = 0x48;
    const POISON: u32 = 0xBAD;

    let hle = HleRegistry::new();
    hle.register("libtest", "sceTestCallGuestExit", hle_call_guest_exit_probe);

    let mut image = vec![0u8; 0x60];
    let mut off = 0usize;
    emit_lea_rip(&mut image, &mut off, 7, CB); // rdi = &callback
    emit_indirect_call(&mut image, &mut off, PROBE_SLOT);
    image[off] = 0xB8; // mov eax, POISON (must never execute)
    image[off + 1..off + 5].copy_from_slice(&POISON.to_le_bytes());
    image[off + 5] = 0xC3;
    // callback: call __stack_chk_fail, then a poison tail that must never run.
    let mut cb = CB;
    image[cb..cb + 4].copy_from_slice(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8
    cb += 4;
    emit_indirect_call(&mut image, &mut cb, CHK_SLOT);
    image[cb] = 0xB8;
    image[cb + 1..cb + 5].copy_from_slice(&POISON.to_le_bytes());
    image[cb + 5] = 0xC3;
    image[PROBE_SLOT..PROBE_SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    image[CHK_SLOT..CHK_SLOT + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + 8).to_le_bytes());

    let linked = callback_fixture_module(
        image,
        &[
            ("libtest", "sceTestCallGuestExit"),
            ("libc", "__stack_chk_fail"),
        ],
    );
    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("a fatal unwind under a callback is a reported exit, not a host fault");
    assert_eq!(
        result,
        raeen_hle::STACK_CHK_FAIL_EXIT_CODE,
        "the run must end with the canary-smash exit code from under the callback"
    );
    assert!(
        !EXITING_CB_HANDLER_RESUMED.load(std::sync::atomic::Ordering::SeqCst),
        "the interrupted HLE handler must never resume after the unwind"
    );
}

fn hle_direct_gateway_call_guest_probe(ctx: &HleContext, _args: &[u64]) -> u64 {
    // A nonzero, genuinely-executable entry: the refusal must fire BEFORE any
    // attempt to run it (were it dispatched, the probe would return 0xE3).
    match ctx.guest_calls.call_guest(GUEST_ARENA_BASE, [0; 6]) {
        Err(raeen_hle::GuestCallError::Unsupported) => 0xE1,
        Err(_) => 0xE2,
        Ok(_) => 0xE3,
    }
}

/// The direct leaf gateway cannot host synchronous guest re-entry (its
/// generated bridge re-bases RSP to a fixed host stack top on every entry):
/// `call_guest` must refuse loudly with `Unsupported` on that path rather
/// than corrupt the gateway frames.
#[test]
fn call_guest_is_refused_on_the_direct_gateway_path() {
    if !fsgsbase_available() || std::env::var_os("RAEEN_DISABLE_DIRECT_HLE").is_some() {
        return;
    }
    let hle = HleRegistry::new();
    // Override a direct-dispatchable leaf with a probe that attempts a
    // synchronous callback from inside the gateway.
    hle.register("libc", "strlen", hle_direct_gateway_call_guest_probe);

    let result = execute_linked(
        &direct_import_module("libc", "strlen"),
        &hle,
        &OrbisKernel::new(),
        0,
        &[],
    )
    .expect("the refused callback must not disturb the direct dispatch itself");
    assert_eq!(
        result, 0xE1,
        "call_guest inside the direct gateway must report GuestCallError::Unsupported"
    );
}

/// End-to-end consumer: guest `qsort` with a REAL guest comparator. The
/// entry sorts a 6-element u64 array through the libc HLE, then verifies the
/// order IN GUEST CODE and returns the comparator's own call counter — so a
/// pass proves the comparator executed (counter ≥ n-1), received real
/// element pointers (order is right), and the array memory was really moved.
#[test]
fn qsort_sorts_a_guest_array_with_a_guest_comparator() {
    const CMP: usize = 0x60;
    const SLOT: usize = 0xA0;
    const COUNTER: usize = 0xA8;
    const ARRAY: usize = 0xB0;
    const VALUES: [u64; 6] = [5, 1, 4, 2, 6, 3];

    let hle = HleRegistry::new();
    let mut image = vec![0u8; 0x100];
    let mut off = 0usize;
    // qsort(&array, 6, 8, &comparator)
    emit_lea_rip(&mut image, &mut off, 7, ARRAY); // rdi
    image[off..off + 5].copy_from_slice(&[0xBE, 0x06, 0x00, 0x00, 0x00]); // mov esi, 6
    off += 5;
    image[off..off + 5].copy_from_slice(&[0xBA, 0x08, 0x00, 0x00, 0x00]); // mov edx, 8
    off += 5;
    emit_lea_rip(&mut image, &mut off, 1, CMP); // rcx
    emit_indirect_call(&mut image, &mut off, SLOT);
    // Verify ascending order in guest code: rsi walks, ecx counts pairs.
    emit_lea_rip(&mut image, &mut off, 6, ARRAY); // rsi
    image[off..off + 5].copy_from_slice(&[0xB9, 0x05, 0x00, 0x00, 0x00]); // mov ecx, 5
    off += 5;
    let check = off;
    image[off..off + 3].copy_from_slice(&[0x48, 0x8B, 0x06]); // mov rax, [rsi]
    off += 3;
    image[off..off + 4].copy_from_slice(&[0x48, 0x39, 0x46, 0x08]); // cmp [rsi+8], rax
    off += 4;
    let jb_fail = off; // jb fail (patched below)
    image[off] = 0x72;
    off += 2;
    image[off..off + 4].copy_from_slice(&[0x48, 0x83, 0xC6, 0x08]); // add rsi, 8
    off += 4;
    image[off..off + 2].copy_from_slice(&[0xFF, 0xC9]); // dec ecx
    off += 2;
    image[off] = 0x75; // jnz check
    image[off + 1] = (check as i64 - (off as i64 + 2)) as i8 as u8;
    off += 2;
    // mov eax, [rip+COUNTER]; ret
    image[off..off + 2].copy_from_slice(&[0x8B, 0x05]);
    let counter_disp = (COUNTER as i64 - (off as i64 + 6)) as i32;
    image[off + 2..off + 6].copy_from_slice(&counter_disp.to_le_bytes());
    off += 6;
    image[off] = 0xC3;
    off += 1;
    let fail = off;
    image[jb_fail + 1] = (fail as i64 - (jb_fail as i64 + 2)) as i8 as u8;
    image[off] = 0xB8; // mov eax, 0xBAD (order violated)
    image[off + 1..off + 5].copy_from_slice(&0xBADu32.to_le_bytes());
    image[off + 5] = 0xC3;
    assert!(off + 6 <= CMP, "entry code must not overlap the comparator");

    // Comparator(a=rdi, b=rsi): count the call, then C-style <0/0/>0 on the
    // pointed-at u64s.
    let mut cmp = CMP;
    image[cmp..cmp + 2].copy_from_slice(&[0xFF, 0x05]); // inc dword [rip+COUNTER]
    let inc_disp = (COUNTER as i64 - (cmp as i64 + 6)) as i32;
    image[cmp + 2..cmp + 6].copy_from_slice(&inc_disp.to_le_bytes());
    cmp += 6;
    image[cmp..cmp + 3].copy_from_slice(&[0x48, 0x8B, 0x07]); // mov rax, [rdi]
    cmp += 3;
    image[cmp..cmp + 3].copy_from_slice(&[0x48, 0x3B, 0x06]); // cmp rax, [rsi]
    cmp += 3;
    image[cmp..cmp + 5].copy_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1
    cmp += 5;
    image[cmp..cmp + 2].copy_from_slice(&[0x77, 0x09]); // ja done
    cmp += 2;
    image[cmp..cmp + 5].copy_from_slice(&[0xB8, 0xFF, 0xFF, 0xFF, 0xFF]); // mov eax, -1
    cmp += 5;
    image[cmp..cmp + 2].copy_from_slice(&[0x72, 0x02]); // jb done
    cmp += 2;
    image[cmp..cmp + 2].copy_from_slice(&[0x31, 0xC0]); // xor eax, eax
    cmp += 2;
    image[cmp] = 0xC3; // done: ret
    assert!(cmp < SLOT, "comparator must not overlap the import slot");

    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());
    for (i, v) in VALUES.iter().enumerate() {
        image[ARRAY + i * 8..ARRAY + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }

    let linked = callback_fixture_module(image, &[("libc", "qsort")]);
    let kernel = OrbisKernel::new();
    let result = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect("qsort with a guest comparator must complete");
    assert_ne!(result, 0xBAD, "the array must be ascending after qsort");
    assert!(
        (5..=64).contains(&result),
        "the guest comparator must have run a plausible number of times \
         (n-1 ≤ calls ≤ 64 for n=6), got {result}"
    );
}

/// Where [`hle_host_fault_probe`] dereferences. A runtime-loaded value, not a
/// literal `0`: the point is to make the *hardware* fault, and a literal null
/// dereference is UB the optimizer is free to turn into `ud2` (or delete)
/// instead of emitting the load this test needs.
static HOST_FAULT_ADDRESS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0x40);

/// An HLE handler with an emulator bug in it — the exact shape the VEH used to
/// misreport: a bad dereference in *host* Rust code while a guest call is in
/// flight.
fn hle_host_fault_probe(_ctx: &HleContext, _args: &[u64]) -> u64 {
    let address = HOST_FAULT_ADDRESS.load(std::sync::atomic::Ordering::SeqCst);
    // SAFETY: none — deliberately. This models the defect under test (an HLE
    // handler dereferencing a pointer that is not valid host memory) so the
    // runtime's classification of the resulting access violation can be
    // observed. The read faults; control never returns from it, because the VEH
    // recognizes the faulting Rip as host-owned and long-jumps to the run's
    // recovery point, which is precisely the behaviour being pinned.
    unsafe { (address as *const u64).read_volatile() }
}

/// An access violation raised inside a Rust HLE handler is **our** bug, and must
/// be reported as one.
///
/// It used to be recorded as `RuntimeError::Faulted { addr: <host rip> }` —
/// indistinguishable from a title dereferencing a wild pointer, so a crash
/// report read "guest fault at 0x7ff…" and sent the investigation to the guest.
/// The host verdict must name the host Rip and the HLE call that was executing.
#[test]
fn an_access_violation_inside_an_hle_handler_is_reported_as_a_host_fault() {
    const SLOT: usize = 0x40;

    let hle = HleRegistry::new();
    hle.register("libtest", "sceTestHostFault", hle_host_fault_probe);

    let mut image = vec![0u8; 0x80];
    let mut off = 0usize;
    emit_indirect_call(&mut image, &mut off, SLOT);
    image[off] = 0xC3; // ret
    image[SLOT..SLOT + 8].copy_from_slice(&HLE_TRAMPOLINE_BASE.to_le_bytes());

    let linked = callback_fixture_module(image, &[("libtest", "sceTestHostFault")]);
    let kernel = OrbisKernel::new();
    let err = execute_linked(&linked, &hle, &kernel, 0, &[])
        .expect_err("a host fault must end the run, not return a value");

    match err {
        RuntimeError::HostFaulted {
            rip,
            access,
            kind,
            hle,
        } => {
            assert_eq!(
                access, 0x40,
                "the reported access address is what host code touched"
            );
            assert_eq!(kind, raeen_runtime::FaultKind::Read);
            // Outside the arena's whole 2 TiB reservation — the host image
            // happens to sit *above* the 16 TiB arena base, so "not a guest
            // address" is a range test, not a comparison.
            assert!(
                !(GUEST_ARENA_BASE..GUEST_ARENA_BASE + 0x200_0000_0000).contains(&rip),
                "the faulting Rip must be the host code's, not a guest address: {rip:#x}"
            );
            assert_eq!(
                hle.as_deref(),
                Some("libtest::sceTestHostFault"),
                "a host fault must name the HLE call that was in flight"
            );
        }
        RuntimeError::Faulted { addr, access, kind } => panic!(
            "an emulator-side access violation was laundered as a guest fault \
             (addr {addr:#x}, {kind} of {access:#x}) — this is the defect"
        ),
        other => panic!("expected Err(HostFaulted {{ .. }}), got {other:?}"),
    }
}

/// A guest worker that ends via `scePthreadExit` while still holding a mutex
/// must have that mutex released.
///
/// Lock release used to run only when `dispatch::run` returned `Err` — but
/// `scePthreadExit` ends a worker with `Ok(Returned)`, so a thread that exited
/// from inside a critical section (a C++ worker that throws, catches, and exits)
/// left the mutex owned forever and every later waiter blocked on it
/// permanently. The worker here never unlocks, so an owner of 0 afterwards can
/// only come from thread-death recovery.
///
/// Deterministic by construction: `scePthreadJoin` joins the host worker, whose
/// closure performs the release before it finishes, and the main guest thread
/// only then calls `exit`. No sleeps, no polling.
#[test]
fn a_worker_exiting_via_pthread_exit_releases_the_mutex_it_still_held() {
    const WORKER: usize = 0x100;
    const MUTEX: usize = 0x170;
    const THREAD_OUT: usize = 0x180;
    const RETVAL_OUT: usize = 0x188;
    const CREATE_SLOT: usize = 0x1C0;
    const JOIN_SLOT: usize = 0x1C8;
    const EXIT_SLOT: usize = 0x1D0;
    const LOCK_SLOT: usize = 0x1D8;
    const PTHREAD_EXIT_SLOT: usize = 0x1E0;

    let hle = std::sync::Arc::new(HleRegistry::new());
    let kernel = std::sync::Arc::new(OrbisKernel::new());
    let mut image = vec![0u8; 0x300];

    let thread_out = GUEST_ARENA_BASE + THREAD_OUT as u64;
    let retval_out = GUEST_ARENA_BASE + RETVAL_OUT as u64;
    let worker = GUEST_ARENA_BASE + WORKER as u64;
    let mutex = GUEST_ARENA_BASE + MUTEX as u64;

    // main: create(worker) -> join(thread, &retval) -> exit(0)
    let mut off = 0;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBF]); // mov rdi, thread_out
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 2].copy_from_slice(&[0x31, 0xF6]); // xor esi, esi (attr = 0)
    off += 2;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBA]); // mov rdx, worker
    image[off + 2..off + 10].copy_from_slice(&worker.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x31, 0xC9]); // xor rcx, rcx (arg = 0)
    off += 3;
    emit_indirect_call(&mut image, &mut off, CREATE_SLOT);
    image[off..off + 2].copy_from_slice(&[0x48, 0xA1]); // mov rax, [thread_out]
    image[off + 2..off + 10].copy_from_slice(&thread_out.to_le_bytes());
    off += 10;
    image[off..off + 3].copy_from_slice(&[0x48, 0x89, 0xC7]); // mov rdi, rax
    off += 3;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBE]); // mov rsi, retval_out
    image[off + 2..off + 10].copy_from_slice(&retval_out.to_le_bytes());
    off += 10;
    emit_indirect_call(&mut image, &mut off, JOIN_SLOT);
    image[off..off + 2].copy_from_slice(&[0x31, 0xFF]); // xor edi, edi
    off += 2;
    emit_indirect_call(&mut image, &mut off, EXIT_SLOT);

    // worker: scePthreadMutexLock(&mutex) then scePthreadExit(0) — never unlocks.
    off = WORKER;
    image[off..off + 2].copy_from_slice(&[0x48, 0xBF]); // mov rdi, mutex
    image[off + 2..off + 10].copy_from_slice(&mutex.to_le_bytes());
    off += 10;
    emit_indirect_call(&mut image, &mut off, LOCK_SLOT);
    image[off..off + 2].copy_from_slice(&[0x31, 0xFF]); // xor edi, edi
    off += 2;
    emit_indirect_call(&mut image, &mut off, PTHREAD_EXIT_SLOT);
    image[off] = 0xC3; // ret (unreachable)

    for (slot, index) in [
        (CREATE_SLOT, 0u64),
        (JOIN_SLOT, 1),
        (EXIT_SLOT, 2),
        (LOCK_SLOT, 3),
        (PTHREAD_EXIT_SLOT, 4),
    ] {
        image[slot..slot + 8].copy_from_slice(&(HLE_TRAMPOLINE_BASE + index * 8).to_le_bytes());
    }

    let linked = std::sync::Arc::new(LinkedModule {
        image,
        base: GUEST_ARENA_BASE,
        executable_ranges: Vec::new(),
        unresolved: Vec::new(),
        unresolved_stubs: Vec::new(),
        module_inits: Vec::new(),
        hle_trampolines: vec![
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadCreate".into(),
                addr: HLE_TRAMPOLINE_BASE,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadJoin".into(),
                addr: HLE_TRAMPOLINE_BASE + 8,
            },
            HleTrampoline {
                library: "libc".into(),
                function: "exit".into(),
                addr: HLE_TRAMPOLINE_BASE + 16,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadMutexLock".into(),
                addr: HLE_TRAMPOLINE_BASE + 24,
            },
            HleTrampoline {
                library: "libkernel".into(),
                function: "scePthreadExit".into(),
                addr: HLE_TRAMPOLINE_BASE + 32,
            },
        ],
        entry: 0,
        tls: None,
        tls_layout: Vec::new(),
        procparam_offset: None,
        unwind_modules: Vec::new(),
    });

    let outcome = execute_process_shared(
        linked,
        hle,
        std::sync::Arc::clone(&kernel),
        &["/app0/eboot.bin"],
        &[],
    )
    .expect("the process must end cleanly through exit(0)");
    assert_eq!(outcome, RunOutcome::Exited(0));

    let state = kernel
        .pthread_mutexes
        .get(&mutex)
        .expect("the worker's implicit mutex creation must be visible");
    let held = state.state.lock();
    assert_eq!(
        held.owner, 0,
        "a worker that exited via scePthreadExit inside a critical section must \
         not leave the mutex owned — every later waiter would block forever"
    );
    assert_eq!(held.recursion, 0);
}
