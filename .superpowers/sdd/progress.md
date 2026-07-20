- ASTRO.BOT boot stall DIAGNOSED (2026-07-19) — it is NOT a missing symbol and
  NOT a deadlock in our primitives; it is the title's own job system never being
  woken. Three reusable diagnostics were added to get here:
  * `sceKernelCreateEventFlag` now logs the flag's **name**. That immediately
    identified every flag ASTRO parks on as AUDIO: `sndz_stream_task_notify`
    (h=2), `SceSndzRenderNotify` (h=4), `SceSndzUpdateNotify`,
    `SceSndzAudioOutBounce*`. Those waits return ETIMEDOUT (0x8002003c) on the
    50 ms slice forever — idle audio, NOT the boot blocker.
  * STALL_DUMP now includes the **guest console tail**. A hung title never
    reaches the end-of-run console dump, so its own log was invisible. ASTRO's
    says it is in `Setup MaterialPackedShaderBinaries` / `Material [...] is
    Replaced` — engine material+shader setup — and the console is **frozen at
    3086 bytes across 8 dumps over 200 s**, so it is genuinely stuck there, not
    progressing slowly.
  * `XPS5X_TRACE_SPIN=1` reports the guest caller of each `sceKernelUsleep`
    once. ASTRO's main thread spins at **module+0xdd8e40**:
    `cmp qword [rbp-0x5d8],0; jne exit; call usleep(1ms); call module+0xdefa40;
    test eax,eax; jns loop` — i.e. *sleep until a worker sets my completion
    flag*, polling an internal (non-import) helper.
  * SIGNATURE: 103 `scePthreadCondWait` vs **1** `scePthreadCondSignal` and 0
    broadcasts — the worker pool parks and is never woken, while the producer
    (t1) sleeps waiting for it. 44 guest threads live. This is the SAME shape as
    the open Dragon Ball task-drain (task #6), so it is likely one root cause.
  * RULED OUT: async IO (no `sceKernelAio*` calls at all; 12 opens/5 reads all
    synchronous) and VideoOut (only `sceVideoOutOpen` — the title never reaches
    a flip/vblank, so the display path is not what it waits on).
  * NEXT: find who is supposed to signal that condvar / set `[rbp-0x5d8]` —
    trace the guest job-post path from `module+0xdefa40`, and check whether a
    worker is blocked in a wait we satisfy incorrectly rather than never.

- STALL CLASS NARROWED — new cross-call condvar-starvation diagnostic names the
  blocked threads for the first time (2026-07-19; hle 280 green, clippy+fmt clean).
  * NEW DIAGNOSTIC (`pthread_cond::note_wait_outcome`, `XPS5X_TRACE_COND`): the
    pre-existing >3s check could never fire, because an infinite `cond_wait`
    deliberately returns after a 10 ms slice as a permitted spurious wakeup — so
    no single call is ever long. Starvation is a *streak* of calls that never
    observe a generation change. Now tracked per `(cond, thread)` and reported
    once, with the guest thread's NAME.
  * UNTIL DAWN (UE5) — the sharpest picture yet. 31 named guest threads incl. the
    full RHI set (`AgcSubmissionThread`, `AgcCleanupThread`, `AgcInterruptThread`),
    `IoDispatcher`, `IoService`, IOThreadPool, Foreground/Background workers.
    STARVED (~650 re-waits each, ZERO genuine wakes): main(t1) on 0x10081375a58,
    `AgcSubmissionThread` on 0x1008137a978, `AgcCleanupThread` on 0x1008137b488,
    `IoDispatcher`, `SystemEventGatherThreadSony`, `OutputDeviceRedirector`.
    Main's event is in the SAME allocation block as the two AGC events.
    **NO `RenderThread`/`RHIThread` exists, and ZERO libSceAgc/VideoOut calls are
    ever made** — UE5 starts the render thread during engine init, so init never
    completes. The engine created its RHI threads and then stopped before using
    the GPU API at all.
  * RULED OUT this round (all verified correct, not the bug): `cond_wait` DOES
    release the guest mutex (`mutex_unlock_for_cond`) and reacquire; `WaitEqueue`
    caps unbounded waits and returns a timeout so the guest re-waits;
    `sceKernelStat` ENOENTs are ~15/709 (normal UE optional-file probing); the
    missing `.uproject` is cosmetic (cooked UE titles don't ship it).
  * LATENT HAZARD FOUND: `libkernel.rs` still defines no-op `hle_pthread_cond_wait`
    /`_signal` stubs that are silently shadowed only because `pthread_cond::register`
    runs AFTER `libkernel::register` and `HleRegistry::register` is last-write-wins.
    Reordering registration would deadlock every title. Delete the dead stubs.
  * ASTRO.BOT equivalent: stalls in `Setup MaterialPackedShaderBinaries`, main
    spins at module+0xdd8e40 polling internal dispatcher module+0xdefa40
    (id space 0x1100xxxx, error 0x8a84006e — in no reference catalogue).
  * NEXT: find what signals main's event. For Until Dawn that means tracing UE5's
    PS5 RHI init handshake — which thread is expected to post to
    cond 0x10081375a58 before `StartRenderingThread`.

- UNTIL DAWN boot traced deep into UE engine init (2026-07-19; workspace tests
  17 suites / 0 failures, touched-crate clippy + fmt clean):
  * It is NOT stuck on content discovery. With `XPS5X_LOG="warn,xps5x_hle::libkernel=debug"`
    it touches **647 distinct `/app0` paths** — the whole UE config cascade
    (engine/project/platform `.ini`s, DataDrivenPlatformInfo, plugins, mods,
    internationalization, localization) — so the VFS, path mapping and file IO
    are working. The 92 GB install is complete (`bates/content/paks/` holds
    `global.utoc`, `pakchunk0-ps5.{pak,ucas,utoc}`, …).
  * The 8 failing opens are all OPTIONAL probes (`bates/saved/paks`,
    `engine/content/paks`, `bates/intermediate/shaders/tmp`, `deepfiles`,
    `/app0/..`) — normal UE discovery, not the blocker.
  * `sceKernelAllocateMainDirectMemory` returns EAGAIN 18× — also NOT a bug:
    `hle_allocate_direct_memory` documents that titles "allocate in a loop until
    ENOMEM and size their pools from the total". The shrinking retry ladder
    (0xd6000000 → 0xd600000 → 0xd60000 → 0xd0000) is that probe. ~13.8 GiB of
    the 14.3 GiB pool is handed out before the failures start.
  * FIXED along the way (real defects, neither was the blocker):
    `sceKernelAvailableDirectMemorySize` reported the whole window and ignored
    `direct_memory_allocated`, so it over-reported free memory to a title sizing
    its pools; and `scePthreadCondDestroy` was bound to a no-op sharing
    `CondInit`'s handler, leaving destroyed conds' state (and waiter generation)
    behind — now bound to the real `hle_cond_destroy`.
  * REMAINING BLOCKER (unsolved): after the config cascade every thread parks.
    Main + `AgcSubmissionThread` + `AgcCleanupThread` + `IoDispatcher` +
    `SystemEventGatherThreadSony` all starve on distinct conds, all entered
    through the same wait wrapper at module+0x1236d, all starting at the same
    moment (~548 waits / 8 s each). No `RenderThread` is ever created and no
    libSceAgc/VideoOut call is ever made. So UE aborts or blocks somewhere
    between config load and `RHIInit`/`StartRenderingThread`.
  * STACK WALK DONE (`guest_stack_chain` in pthread_cond.rs, in the STARVED
    report): each starved waiter's engine-level frames are now recovered.
    Main thread: `+0x1236d <- … <- +0x18d4ebd <- … <- +0x8ef82b`.
    AgcSubmissionThread: `… <- +0x4f2f1c0 <- +0x74ee878`.
    AgcCleanupThread:   `… <- +0x4f30055 <- +0x6360f88`.
    Background workers: `… <- +0x1b2eb <- +0x74d3c90 <- +0x1ae41 <- +0x74d8600`.
    Decoding main's frame at module+0x18d4ebd shows two `mov ecx,4; mov rdi,r15;
    call module+0x407a0` sites — and 0x407a0 and 0x1236d are both UE's OWN
    wrappers (real prologues, not PLT thunks), so the chain stays inside UE's
    event machinery rather than reaching an emulator boundary.
  * HONEST STATUS: four levels of RE in, every frame is UE-internal and no
    emulator gap is implicated. Cracking this needs UE5 source/symbols or a
    much longer RE effort — it is NOT a missing-import or broken-primitive
    problem. Do not re-derive the above; start from these frames.

# XPS5X session progress ledger

- MILESTONE: **all 5 installed titles load and execute with ZERO unresolved
  imports** (2026-07-19). Workspace `cargo test` 0 failures; touched-crate
  clippy + fmt clean. Every title previously died on a missing symbol.
  * ASTRO.BOT additionally cleared its whole `sce::Json` boot path. Implemented
    `libsce_json.rs` properly rather than as stubs: `Value` and `String` payloads
    live in host-side maps keyed by the guest `this` pointer (the C++ layout is
    opaque, so a ctor/`set` that only returned `this` would leave the guest
    reading garbage back out). Covers Value ctors + `set` for
    bool/long/ulong/double/`const char*`/`const Value&`/`const String&`/
    `ValueType`, String ctor/dtor, and both Itanium C1/C2 + D1/D2 variants.
    `set(double)` is the first real consumer of the new float-argument channel.
    Then: Share content-event callbacks and the notice-screen skip flags.
  * **HONEST LIMIT — this is "loads and runs", NOT "boots to a menu."** All five
    survive a 45-50 s bounded run instead of dying, but they then sit: Until Dawn
    and Dragon Ball park in `sceKernelWaitEqueue`, and STALL_DUMP reports 0
    registered guest threads at dump time even for titles observed creating 18-20
    threads earlier in the same run. Nothing here demonstrates a rendered frame or
    a reached menu. The next frontier is the WAIT/STALL class (tasks #6/#8) —
    why the event-queue waits never wake — not missing symbols.


- ALL-TITLES boot sweep — 4 of 5 installed games now load+execute with **zero
  unresolved imports** (2026-07-19; hle 279, firmware 108/10/3/3, runtime 48/43
  green; touched-crate clippy --no-deps + fmt clean; re-measured after every fix):
  * Method: run every installed eboot headlessly (`--run-eboot`, 45 s bound),
    read the reported unresolved import, fix, rebuild, re-measure. Twelve rounds.
  * **PROVIDER-ALIAS BUG CLASS — the dominant finding.** Resolution is
    provider-aware (`ModuleRegistry::resolve` keys on the importing symbol's
    library), and several functions were **already implemented** but registered
    under a library no title names. Each cost a game its boot:
    - `gettimeofday`/`usleep` — registered `libScePosix`, imported `libkernel`
      (Minecraft). Joined `clock_gettime`, fixed earlier the same way.
    - `sceShareInitialize`/`sceShareSetContentParam` — registered
      `libSceShareUtility`, imported `libSceShare` (Dragon Ball). An earlier pass
      had DELETED the `libSceShare` spelling as a "wrong name"; it is real.
    - `in6addr_any`/`in6addr_loopback` — the HLE data page registered ALL its
      exports under `libkernel`, but Minecraft imports the IPv6 constants from
      `libSceNet`. Now registered under both. (The stale comment there claimed
      "resolve is by NID and ignores the declaring module" — it is not.)
    - `sceAgcDriverSubmitAcb`/`sceAgcDriverAddEqEvent` — `libSceAgc` vs
      `libSceAgcDriver` (ASTRO.BOT).
  * New implementations (each measured as a title's live blocker): libc
    `_init_env`, `__cxa_guard_acquire/release/abort` (Itanium static guards,
    host-mutex + condvar so two guest threads cannot double-construct a static),
    `strcspn`, `wcslen`, `wcscpy`, `strtok`, `vsnprintf` (with a real SysV
    `va_list` walker — `GuestVaList` — reusable for the whole `v*printf`
    family), `sincosf`; libkernel `scePthreadMutexattrSetprotocol`,
    `sceKernelMapNamedFlexibleMemory` (the public spelling was missing while
    `...Internal` existed), `sceKernelAvailableDirectMemorySize`,
    `sceKernelIsTrinityMode`; `sceSystemServiceGetHdrToneMapLuminance`;
    `sceNpWebApi2PushEventCreateHandle` (refuses, per this module's documented
    offline policy); libSceRudp init/event-handler/IO-thread; `sce::Json::Value`
    ctor/dtor.
  * **NEW RUNTIME CAPABILITY — floating-point arguments.** HLE handlers only ever
    saw integer registers, so any function taking a `float`/`double` (SysV passes
    them in XMM0-7) was unimplementable without guessing. Added
    `HleContext::float_args` (low 64 bits of XMM0..XMM7, filled from the trap
    CONTEXT's `FltSave.XmmRegisters`) plus `float_arg_f32/f64` helpers. Two
    construction sites, no change to the `HleFunction` signature. `sincosf` is
    the first consumer; the whole libm surface needs it.
  * **STATE (honest).** Minecraft, Until Dawn, A Plague Tale Requiem and Dragon
    Ball all now run the full 45 s with no unresolved import — but they are
    **stalling, not finishing a boot**: Until Dawn and Dragon Ball park in
    `sceKernelWaitEqueue`, Minecraft's last call is `pthread_mutexattr_destroy`,
    and none spawn many threads. Clearing the import walls moved them from
    "dies instantly" to "runs and waits"; the remaining work is the WAIT/stall
    class (tasks #6/#8), not missing symbols.
  * **ASTRO.BOT** went furthest: past AGC init, **18 guest threads**, now blocked
    on the `sce::Json` C++ library — 45 distinct mangled imports (Value ctors for
    every type, `set*`, `referArray/Object/Value`, `serialize`, Array +
    iterators, Object, String). That is a library port, not a stub: doing it
    honestly needs a host-side object model keyed by the guest `this` pointer so
    values round-trip instead of reading back garbage. Tracked as task #14.


- ASTRO.BOT AGC wall — 4 blockers cleared, title now reaches shader creation
  (2026-07-19, working tree; hle 277 + firmware 108/9/3/3 green, clippy+fmt
  clean; measured against the retail title after each fix):
  * **Repeatable technique established** for unknown NIDs: (a) recover the name
    from SharpEmu's `aerolib.bin` catalogue — format is
    `[u8 len][encoded-NID][u16 len][name]`, built from `scripts/ps5_names.txt`,
    so `grep -a -o -P '<encodedNid>.{0,50}'` yields the name; (b) get the caller
    from the runtime's guest-stack return-addr chain; (c) `--dump-vaddr` the
    return address to read the ABI off the post-call instructions;
    (d) `--resolve-got` any PLT thunk to name what the branch calls next.
  * `sceAgcGetIsTrinityMode` (NID 0x05f0436466ed8bb0, name recovered from
    aerolib; implemented by NO reference emulator). "Trinity" = PS5 **Pro**;
    XPS5X emulates a base PS5 → returns 0. ABI proven from the call site:
    `test eax,eax; jnz` on the return, so the flag comes back **directly in
    EAX**, no out-param. The ZERO branch is the one that goes on to call
    `sceAgcDriverSubmitDcb` — i.e. base-PS5 is the GPU-submitting path.
  * **Two provider-alias bugs** (same class as the clock_gettime fix):
    `sceAgcDriverSubmitAcb` (0x812467afbf45f2d4) and `sceAgcDriverAddEqEvent`
    (0xc36ac98660fe76c1) were IMPLEMENTED but registered only under `libSceAgc`,
    while Gen5 retail imports them from **`libSceAgcDriver`** — provider-aware
    resolution left them unreachable. Bound both retail identities explicitly
    (the file already had this precedent for SubmitDcb).
  * Unnamed `dolOmWH+huQ` (0x76894e9961fe86e4) — in no catalogue. Call-site ABI:
    `f(void* out, a, b)`; caller pre-zeroes the 16-byte `out`, IGNORES the
    return, reads `*out` and branches on NULL, and that NULL branch is graceful
    (zero-fills the owning struct and continues). So it reports "no object":
    returns 0, leaves `*out` as zeroed, warns once. Deliberately does NOT
    fabricate a handle. NOTE `hle_unknown_filler` (used for `qj7QZpgr9Uw`) is
    command-buffer-specific and would have corrupted this caller's local.
  * Tests: provider-aware `ModuleRegistry::resolve` assertions for the AGC
    identities (`sce_agc_get_is_trinity_mode_...`,
    `agc_driver_entry_points_resolve_from_the_libsceagcdriver_provider`).
  * **Measured trajectory**: first-AGC-import → event-queue registration
    (AddEqEvent ×many) → **`sceAgcCreateShader`**. NEXT WALL: unnamed
    `fd5Bp5tGTgo` (0x7dde41a79b464e0a, libSceAgc), then the Cb/Acb/Dcb builder
    batch (task #13).


- ASTRO.BOT boot chain advanced — B0 verified + Slice 2 (2026-07-19, working
  tree; hle 108 + firmware 118 green incl. new provider-aware clock_gettime
  tests; release GUI rebuilt; measured against the retail title twice):
  * **B0 VERIFIED on the real title** (Slice 1 init-once): a default release
    `--run-eboot` (NO `XPS5X_SKIP_MAIN_INIT`) shows deps init once
    (`libSceNpCppWebApi.prx`/`libc.prx: calling dependency module_start`), the
    main initializer `deferred to crt0`, and **no `0x7426c00` cycle** — the
    title advances past init in ~4 s. The Slice 1 exit gate is closed.
  * **Slice 2 done — `clock_gettime`**: the title imports it (NID
    0x94b313f6f240724d) naming provider library **`libkernel`**, but it was
    registered only under `libScePosix`; resolution is provider-aware, so it
    was unresolved. Registered the existing thin POSIX adapter
    (`posix_clock_gettime`) under `libkernel` too. Tests: hle
    `clock_gettime_is_also_registered_under_libkernel` + firmware
    `clock_gettime_resolves_from_the_libkernel_provider_the_title_names`
    (through the real provider-aware `ModuleRegistry::resolve`, not a
    provider-blind NidDatabase lookup).
  * **Task #7 (ASTRO.BOT "libc allocator mutex spin") does NOT reproduce** on
    the current build: the title progresses through the libc mutex/allocator
    traffic (last HLE calls are scePthreadMutexLock/Unlock/Init) in ~5 s and
    returns cleanly — no spin. That wall (from an older architecture branch) is
    gone.
  * **NEW ASTRO.BOT wall = libSceAgc command-buffer builders** (GPU init). The
    title now reaches AGC and stops on the first unimplemented import, an
    UNNAMED NID `0x05f0436466ed8bb0` ("BfBDZGbti7A"), with a batch behind it:
    sceAgcCbSetShRegistersDirect, sceAgcCbDispatchGetSize, sceAgcCbNopGetSize,
    sceAgcAcb{AcquireMem,CopyData,DmaData,WaitRegMem,Push/PopMarker},
    sceAgcDcb{DrawIndexIndirect,GetLodStats,StallCommandBufferParser}, +
    another unnamed (0x7dde41a79b464e0a). Next step is implementing these AGC
    Cb/Acb/Dcb builders (SharpEmu `AgcExports`/Kyty Gen5) — the ASTRO.BOT plan's
    M2/AGC territory. `libsce_agc.rs` already carries 25/48; these are new.


- GPU depth/stencil attachments in the Vulkan pipeline (2026-07-19, working
  tree; xps5x-gpu 120 lib + all integration green incl. 3 new depth/stencil
  tests on real hw (AMD Radeon 760M); gpu clippy --tests + fmt clean; release
  GUI build green; hle 276 + runtime 48+43 unblocked/green):
  * Completed a half-finished depth/stencil refactor a prior agent left the
    tree non-compiling on (broke the whole workspace since hle/runtime depend
    on xps5x-gpu). `render_draw` now returns `DrawOutput { color, depth }`.
  * `offscreen.rs`: `read_back_color`/`read_back_depth` (Option results; None
    for a depth-only z-prepass); `record_and_submit` gained depth image
    transitions (UNDEFINED→DEPTH_STENCIL_ATTACHMENT_OPTIMAL, seed-on-LOAD via
    the upload buffer, →TRANSFER_SRC + copy-out), dynamic-rendering depth +
    stencil attachments, colour-attachment work guarded by `color_output`;
    `Drop` frees the 7 depth resources.
  * Correctness fix (helps the title path too): `VulkanDevice::
    supports_depth_stencil_attachment` queries format features; `create_depth_
    target` fails cleanly on an unsupported format instead of creating a bogus
    image the driver accepts then errors on every use. Measured: AMD exposes
    D32_SFLOAT_S8_UINT but NOT D24_UNORM_S8_UINT — the stencil test tries both.
  * Tests (`tests/depth_stencil.rs`): depth CLEAR readback, depth LOAD (prior
    contents), and stencil CLEAR readback — write disabled so the readback is
    exactly the clear/loaded value, independent of shader z. All pass on hw.
  * Fixed the two broken `render_draw` callers (blend_state/texture_upload
    tests → `.color`) and `draw_translate` (colour-only composite passes
    `color_output: true, depth: None`).
  * OPEN FOLLOW-UP (noted in code): no `depth_state_from_regs` yet — real PM4
    draws still pass `depth: None`, so the attachment is wired+tested but
    DORMANT until the DB_* register decode feeds it (DB_DEPTH_CONTROL / Z_INFO
    / STENCIL_INFO / RENDER_CONTROL, per the `DepthState` doc's Kyty refs) plus
    a depth-framebuffer map keyed by DB_Z_WRITE_BASE.
  * Pre-existing test-clippy debt cleared in-crate while here: `#![allow(
    deprecated)]` on the m2 fixture gate, `#[allow(field_reassign_with_default)]`
    on one nested-array test setup, and an unused `mut` on the compute-seed test.


- ASTRO.BOT boot Slice 1 — process init runs exactly once (2026-07-18,
  working tree; core 7/7, firmware 106/106 lib, runtime 43/43 execute +
  full suite green; touched-crate clippy --no-deps clean; fmt clean; release
  GUI build green; passed a 3-lens adversarial review workflow — 3 confirmed
  findings, all fixed below, 0 runtime-correctness bugs):
  * ROOT CAUSE (measured): retail crt0 `_start` walks the executable's own
    init array itself; the loader ALSO called the main initializer, so
    constructors ran twice — a list-adding ctor then built a cyclic list its
    own walk hung on (t1 at `module+0x7426c00`). `XPS5X_SKIP_MAIN_INIT=1`
    proved the cause; this slice makes the proven path the DEFAULT (no env
    var).
  * `ModuleInitRole::{Dependency, Main}` (firmware `linker.rs`) on
    `ModuleInit`; `load_process` tags dependency initializers Dependency and
    the appended executable DT_INIT Main.
  * `EntryPolicy::{CrtOwnsMainInit, LoaderOwnsMainInit}` (runtime). Shared
    `run_module_initializers` loop: `execute_process` uses CrtOwnsMainInit
    (dependency initializers run, the main one is WITHHELD for crt0);
    `execute_linked` uses LoaderOwnsMainInit (no crt0 -> loader runs every
    initializer, main included; a safe no-op for today's empty-`module_inits`
    callers). `sceKernelLoadStartModule`-owned PRX init unchanged.
  * `XPS5X_SKIP_MAIN_INIT` demoted from mechanism to a one-shot deprecation
    warning; default now takes the proven path.
  * `DiagnosticKind::ModuleInit`: each initializer transition (run / deferred)
    recorded with module name, role, ordinal, and the recorder's stable
    sequence (guest_thread=1) when deterministic diagnostics are enabled.
  * TDD: 4 new runtime tests (dep-runs-once + main-withheld; main-runs-once-
    via-crt0; execute_linked loader-owns-main; diagnostic transition record).
    RED-GREEN PROVEN: temporarily reverting the policy made the two process
    tests fail with main=1 (was 0) and main=2 (was 1), and the diagnostic
    test fail on the role detail; restoring it turned them green.
  * Review fixes: (1) rewrote the now-stale firmware comment that still claimed
    the loader owns the executable's DT_INIT; (2) extracted
    `append_dependency_initializer` + a unit test pinning the Dependency role at
    its single production site (the loop's role assignment was previously
    asserted nowhere); (3) narrowed `EntryPolicy` to `pub(crate)` (it types no
    public signature; `ModuleInitRole` stays pub — it types `ModuleInit.role`).
  * Pre-existing, OUT OF SCOPE: `cargo clippy --workspace -- -D warnings` fails
    on ~a dozen `collapsible_if` sites (nid.rs, dispatch.rs, native_trap.rs,
    thread.rs, hle) — a clippy-1.97 toolchain bump on `if let { if }`, NOT this
    change. Touched-crate `--no-deps` clippy is clean. Flagged as a separate
    cleanup task.
  * EXIT GATE STILL OPEN (needs the user's machine): a default release
    `--run-eboot` of ASTRO.BOT must no longer visit the `0x7426c00` cycle and
    must reach the next guest import (measured to be POSIX `clock_gettime`,
    Slice 2). Unit-tested here; not yet re-measured against the title.

- Architectural consolidation (2026-07-18, commit pending): XPS5X-owned
  time/wait/event/VFS/network/GPU-submission contracts are active on the HLE
  boot path; deterministic process diagnostics sequence HLE, wait/wake, event,
  guest-task ownership, and GPU submission; guest memory now exposes validated,
  executable, and GPU-visible capability types; `GuestProcess` explicitly owns
  the active GPU session and drains it before unmapping guest memory; public
  shader metadata no longer re-exports Kyty types; module resolution adds
  strict `HleOnly`/`LleOnly` beside the existing HLE-first default and
  title-supplied LLE preference. Tests: core 6/6, kernel 20/20, firmware 99/99
  + integration 9/9, HLE 271/271, runtime 86/86, GPU 118/118 + Vulkan
  integration 9/9, GUI 120/120.

(Recreated 2026-07-16 — the previous ledger file was absent from the tree;
per-module authority is `docs/reference-port-ledger.md`.)

  * **Blocking WaitEventFlag** (the hot-spin fix): poll-and-instant-ETIMEDOUT
    was built for single-active-execution; with real guest threads it made 5
    threads spin at 100% CPU. Now a shared (Mutex,Condvar)
    `event_flag_signal` — Set/Clear/Cancel notify_all, WaitEventFlag blocks
    with a deadline (NULL timeout = 50ms forever-slice, equeue-style; bounded
    honors requested µs capped at 50ms). Test: cross-thread
    wait_blocks_until_another_thread_sets_the_flag (263/263 hle).
    Measured: the 5 spinners now park at a host wait; t1(main) still stalls.
  * **Dragon Ball stall root-caused to the guest level** (next gate): t1 spins
    at `eboot+0x2bd2151` — a linked-list walk waiting for every task's
    state field (+0x48) to hit 0 (a TASK-QUEUE DRAIN WAIT, yielding between
    polls via call 0x2bbfc80) while all 13 background workers idle in
    scePthreadCondWait. Tasks never reach the workers: the posting/wake chain
    (job scheduler / event / audio callback) is the next investigation. Not a
    sync-primitive bug anymore — the cond waits are correctly blocked.

- Dragon Ball task-drain stall (2026-07-18, measured via XPS5X_STALL_DUMP +
  --dump-vaddr): the 29 threads are Unreal's TaskGraph (Foreground/Background
  Workers + ThreadPool). t1(main) spins at `eboot+0x2bd2151`: a linked-list
  walk waiting for every task's state field (+0x48) to hit 0 (UE
  WaitUntilTasksComplete), yielding between polls via call 0x2bbfc80 — while
  all 13 workers idle in scePthreadCondWait (correctly blocked; the
  blocking-WaitEventFlag fix parked the earlier hot-spinners). **0 AGC calls
  yet** — the stall is pre-RHI. The task posting/wake chain (or a prerequisite
  task's own gate) is the open question: workers are never woken to consume,
  and the main thread's TLS check (scePthreadGetspecific) never flips. The
  "Project file not found" log is cosmetic — the .uproject is not in
  package.manifest, so a real PS5 logs identically.
  Universal-import note: 359 link-time missing NIDs (libSceAgc Cb/Acb/Dcb
  builders, Audiodec, AudioOut2, AudioIn, NP dialogs) — link-time only, none
  called yet on the boot path (the fault-at-call diagnostic names them as they
  become live).

- Dragon Ball boot chain (2026-07-18, commit pending; 262/262 hle, 20/20
  kernel tests, workspace clippy clean):
  * sceKernelAioInitializeParam/Impl (synchronous AIO model), scePlayGoGetChunkId
    (SharpEmu ABI: count-only vs list mode, BAD_POINTER/BAD_SIZE codes),
    sceKernelAddAmprEvent (equeue registration), pthread_create_name_np
    (create + thread_names record), sceKernelClockGetres ({0,1}),
    sceRtcConvertUtcToLocalTime (real host TZ bias via GetTimeZoneInformation;
    windows-sys added to xps5x-hle + Win32_System_Time feature),
    sceNetCtlRegisterCallbackV6/GetStateV6 (aliases), sceNetEpoll* (offline
    create/control/wait(≤50ms, 0 events)/destroy),
    sceNpWebApi2CreateUserContext (local-user handle).
  * **Full APR/AMPR subsystem**: sceKernelAprResolveFilepathsToIdsAndFileSizes
    (path list → deterministic FNV-1a ids + sizes, missing → 0xFFFFFFFF/0 and
    batch continues; `appr_files` registry in the kernel), AMPR command-buffer
    record append (ReadFile/KernelEventQueue/WriteAddress per SharpEmu
    `AmprExports` layouts), sceKernelAprSubmitCommandBuffer[AndGetResult] +
    WaitCommandBuffer — synchronous completion (real reads via the registry,
    equeue event triggers, address writes, bytesRead backfill).
  * Measured result: the title boots deep — 29 guest pthreads, engine
    allocators up, sysmodules loading (0x10f/0x95/0x96), shader dirs probed.
  * **Stall (XPS5X_STALL_DUMP, next gate)**: t1(main) spins on
    scePthreadGetspecific (waits a TLS flag); t18-22 hot-spin on
    sceKernelWaitEventFlag → instant 0x8002003c (check timeout handling — same
    class as the f258427 WaitEqueue spin); t2 FAP listener polls WaitEqueue
    (audio stub, task #12); 13 workers idle in CondWait. Tracer noise: errno
    heuristic misfires on size-returning sceAmprMeasureCommandSize* (0x30).

- EUD resolver measurements (2026-07-18, XPS5X_TRACE_EUD evidence):
  cs@0x253a5000 (eud_size_dw=8): direct[t5]=s12 → the Region/EUD base pointer
  IS a user SGPR pair: s12:s13 = 0x29b9e0e0 (captured at dispatch). The table
  there holds the SGPR image (dwords match s0..s5 values — s1/s2/s3/s5 exact,
  s0 differs by 0x100, likely a per-draw offset or a different dispatch's
  SGPRs). Its sharp resources: s0+0, s8+1, and s32+0 — s32 is a register whose
  value comes from a scalar load (the two-level chain: EUD table → loaded
  pointer → descriptor). Existing resolver increments: 1
  (`find_scalar_load_bases` — s_load_dword* scan) and 2
  (`scalar_load_target_address` — pair + offset → guest addr). Increment 3
  (fetch EUD table via Region pointer + evaluate constant-offset loads +
  feed `read_sharp_fields(Some(ext))`) is the bounded next step; the VS
  s[14:15] computed-pointer case needs the full evaluator (SharpEmu
  Gen5ShaderScalarEvaluator.cs, 2,362 lines).
  PM4 indirect-register mis-decode: intermittent race (table memory cycles
  between register-pair and descriptor content between submit and CP drain);
  zero repro in 300s; tolerated by the resilience policy; ledgered.

- Hygiene + indirect investigation (2026-07-18, commit pending):
  All workspace clippy errors fixed (hle getdents chain + dead MOUNT_POINT,
  dispatch.rs unused-mut/collapsible-if/range-contains/then-chain,
  launcher needless-borrow, main sort_unstable_by_key) — workspace now
  clippy -D warnings clean, 0 failed test suites.
  "context register index out of range" (vertex data walked as register
  offsets) investigated with XPS5X_TRACE_INDIRECT: **intermittent,
  race-dependent** — zero repro in a 300s run. Evidence says the title's
  indirect-register tables live in its shader user-data memory region (the
  TRACE_EUD SGPR pointers at 0x29b9d520 name the same buffer), which cycles
  between register-pair tables and descriptor tables between submit and CP
  drain. Resilience policy already tolerates it (skip out-of-file writes;
  next submission rewrites them). No fix without a repro — measured, noted.

- EUD-convergence batch (2026-07-18 late, commit pending; 236/236
  kyty-graphics, 115/115 xps5x-gpu, clippy clean):
  * Recompile_Fetch width-mismatch rules, both directions measured (attrib 2
    as 2ch→vec3 fill z=0.0f; as 4ch→vec3 drop w into %temp_float scratch) —
    GCN's (0,0,0,1) default semantics, beyond Kyty (it EXITs on mismatch).
  * Cube textures end-to-end (measured type 11 = 1024x1024x6 skybox, fmt56,
    tile 9): analysis accepts 9/11; spirv.rs emits OpTypeImage Cube from the
    measured T# types (mixed = named error); ImageSample emits vec3 coords for
    cube; decode_texture fetches 6 faces as 6 block grids; offscreen creates
    CUBE_COMPATIBLE image + CUBE view with layer-aware barriers. Tests: cube
    SPIR-V emission + 6-face decode round-trip.
  * input_num = ps_in_control & 0x3f (NUM_INTERP field, AMD layout) — Kyty's
    whole-register read exploded 0x4004 → 16388 bogus truncation; now 4.
  * Shader translation failures converged: **3 remaining, ALL EUD-family**
    (cs eud_size_dw; vs SLoadDwordx2 s14-computed pointer; ps ImageSample
    declined with 0 mapped textures). Next gate is task #4 (EUD/SRT resolver;
    SharpEmu Gen5ShaderScalarEvaluator 2,362 lines; the CS's pointers are
    captured user-SGPRs at dispatch — the tractable first half).

- Shader loop batch (2026-07-18, commit pending; 231/231 kyty-graphics,
  112/112 xps5x-gpu, workspace green, clippy clean; two pre-existing
  clippy errors in xps5x-hle left alone — not this session's code):
  SDWA src abs/neg now PARSE into operand modifiers (beyond Kyty — its vopc
  path exits on any modifier; operand_load_float already applied FAbs/
  FNegate; measured encodings `v_cmp_lt_f32 s2,|v2|,c` / `v_mul_f32 v2,v4,-v3`
  tested). VCmp F32/I32/U32 families wired from staged (Eq/Ne sign-agnostic
  in the I32 family per Kyty's layout; table 223 rows, 211 impl / 12 NI).
  SLoadDwordx2 EUD x2 path (x4/x8 machinery with n=2; the menu VS's s[14:15]
  is a COMPUTED pointer = the real EUD chain — remains open, task #4).
  `%paramN` declarations now cover the body's exp formats (register
  export_count=1 under-read a param1-writing VS — "id %param1 is used but
  never defined" fixed). buffer_load_dwordx4 + offen (beyond Kyty, opcode
  0x0E; voffset adds into the byte address; measured encoding tested).
  naga rejects Kyty's `%_arr_BufferObject_uint_N` pattern (array of struct
  with runtime array) which the driver accepts — that one test asserts on
  assembly+source instead of naga validation.
  Gen5 VB formats mapped per SharpEmu Gfx10UnifiedFormat as the title named
  them: 77 (RGBA32F, 4.5k draws), 71 (RGBA16F, 3.7k), 23 (RG16_UNORM).
  SW_64KB_S (tile 9, Standard 64KiB — AMD/SharpEmu equation) detiler added
  beside SW_64KB_R_X; the 1937x333 atlas decodes (1.9k draws unblocked).
  Non-2D texture measured: type 11 = CUBEMAP 1024x1024x6 fmt56 tile9 (skybox);
  analysis gates now name type/extent/format/tile (NotImplementedOwned).
  Recompile_Fetch: GCN default-fill for attr-narrower fetches (attrib 2 = 2
  channels feeding a vec3 fetch → z = 0.0f; wider stays a named error).
  Run-to-run failure list went 8 distinct → 3 (cs EUD, ps "unknown operand:
  249", vs SLoadDwordx2-EUD-chain); texture-type/ImageSample-decline did not
  reappear in the last two windows.
  Recompile_Fetch: GCN width-mismatch rules both directions (attr narrower →
  (0,0,0,1) default fill; attr wider → dropped channels into a scratch), both
  measured (attrib 2 as 2ch→vec3 and 4ch→vec3 on the menu VS).
  Cube textures (measured type 11 = 1024x1024x6 skybox, fmt56, tile 9):
  analysis accepts 9/11 (others still named with raw fields); SPIR-V emits
  OpTypeImage Cube from the measured T# types (mixed 2D+cube = named error);
  ImageSample coords go vec3 for cube; decode_texture fetches 6 faces as 6
  block grids; offscreen creates CUBE_COMPATIBLE images + CUBE views with
  layer-aware barriers. Tests: cube SPIR-V emission, 6-face decode round-trip.

- Texture chain completed + two GPU blockers (2026-07-18, commit pending;
  106/106 xps5x-gpu lib + 225/225 kyty-graphics, clippy clean):
  * Vulkan consume of `ShaderStageBinding.textures` (the missing half of the
    in-flight texture work): offscreen.rs now builds SAMPLED_IMAGE + SAMPLER
    descriptor arrays per stage (bindings from `shader_calc_binding_indices`),
    uploads guest textures staging→device-local→SHADER_READ_ONLY, one sampler
    per S# (linear/nearest). Acceptance: tests/texture_upload.rs — recompiler-
    ABI PS (push-const indices, OpSampledImage/OpImageSampleImplicitLod)
    samples a magenta texel; center=magenta, corners=clear, 0 validation
    errors. Gotcha: recompiler lists resource vars in OpEntryPoint interface
    (spirv-val requires it under Vulkan 1.3).
  * VB/storage fetch cap bug MEASURED: "vertex buffer guest range not fully
    readable" (7762×/run, only 3 executed draws) was NOT memory — the
    committed-prefix probe printed prefix == size. The 64K-dword
    MAX_READ_DWORDS cap (meant for CP out-of-band mis-decode protection) was
    misapplied to resource fetches; Minecraft binds a 4 MiB vertex-arena V#.
    Fix: `read_dwords_validated` (resource path, 256 MiB cap) for
    draw_translate's read_guest_bytes; the CP out-of-band path keeps the 64K
    cap. Same root cause as run-1's storage-buffer failure.
  * Alpha blending: CB_BLEND0_CONTROL..7 + CB_BLEND_{RED..ALPHA} decode in
    kyty-graphics run.rs (Kyty bit layout; SharpEmu AgcPrimaryRegisterDefaults
    confirms addresses), `blend_state_from_regs` → Vulkan blend attachment +
    constants (dual-source/BOTH_SRC_ALPHA = named errors, never silent ZERO;
    !separate_alpha reuses colour factors = hw behaviour). Acceptance:
    tests/blend_state.rs — SRC_ALPHA/1-SRC_ALPHA/ADD over seeded red reads
    back (128,128,0,128), corner stays seeded, 0 validation errors.
  * Recompiler table tests updated for the ImageSample nine (177 impl/44 NI;
    NI-error test retargeted to ImageStoreMip — no guessed encodings).

- GraphicsRun CommandProcessor (Kyty Gen5 CP): expanded for retail DCBs
  (commit pending, 194/194 kyty-graphics + 86/86 xps5x-gpu tests).
  Resilience policy: unknown op/register = rate-limited warn + skip-by-length;
  hard errors only for truncated/non-type3 streams and refused draws.
  Ported: R_DRAW_INDEX (AGC + IT_DRAW_INDEX_2 raw form), R_{CX,SH,UC}_REGS_INDIRECT
  via new GuestMemory trait, R_DRAW_RESET → Reset, IT_INDEX_TYPE/BASE/BUFFER_SIZE +
  IT_SET_BASE(1) tracking, rate-limited sync/event/write-data skips.
  Indexed/indirect draws degrade to logged vertex-count-only draws
  (DrawSink::draw_index default; indirect count read from first args record).
  xps5x-gpu: guest_mem::IdentityGuestMemory (VirtualQuery-validated identity
  reads) wired into AgcGpuSession::execute_dcb_cp.
  Still todo: GraphicsRender (real index fetch, guest shader bind, multi-draw walk).

- Minecraft (PPSA17221) libkernel + libScePosix import closure: **0 missing**
  in both libraries (was 17 libkernel + 19 libScePosix), measured by re-running
  `--run-eboot`; 144 distinct missing NIDs remain, all in out-of-scope service
  libs (libSceNpWebApi2 21, libSceHttp2 14, libSceNet 13, ...). Implemented in
  xps5x-hle (commit pending; 247/247 hle, 19/19 kernel, 102/102 firmware,
  82/82 runtime tests): real VFS unlink/rmdir/rename/truncate (+ new VFS ops),
  REAL blocking POSIX semaphores (`posix_sem.rs`, address-keyed, condvar +
  termination-aware slices), scePthreadMutexTimedlock (deadline in lock_core),
  sceKernelMapDirectMemory2 (arg reshuffle), Add/DeleteWriteEvent, offline
  POSIX sockets (accept/listen/recv/send/select/... EWOULDBLOCK semantics,
  errno via __error slot), sched_get_priority_max/min (767/256), getrusage
  zero-fill, signal/Mlock/Sync/Chmod/Utimes accepted, `__progname` as a real
  data-page pointer export (xps5x-firmware). Title now boots 17 guest pthreads
  and dies downstream on its own `std::out_of_range` ("invalid string
  position") during phase-1 unwinding — next investigation target.

- ShaderMemory Phase 2 (guest shader fetch → GCN parse → SPIR-V → draw):
  **implemented + proven end-to-end in-tree** (commit pending; 196/196
  kyty-graphics, 87/87 + 2/2 + 2/2 xps5x-gpu tests, clippy clean).
  kyty-graphics CP: Gen5 shader-bind SH registers ported from Kyty's
  g_hw_sh_indirect_func — SPI_SHADER_PGM_LO/HI_PS+CHKSUM_PS+RSRC2_PS,
  PGM_LO/HI_ES+CHKSUM_GS+RSRC2_GS (gs-instead-of-vs), USER_DATA_GS slots —
  plus sh_regs context regs (SPI_SHADER_COL_FORMAT, SPI_PS_INPUT_ENA/ADDR/
  IN_CONTROL, SPI_PS_INPUT_CNTL_0..31, SPI_VS_OUT_CONFIG, DB_SHADER_CONTROL).
  These are exactly the registers Minecraft's DCBs write (proven from the
  prior iron log: unknown-reg warns 0xC8/0xC9/0x80/0x8A/0x8B/0x08, cx 0x191+).
  xps5x-gpu: shader_fetch.rs — bounded fetch (4 KiB chunks, 256 KiB cap,
  parser-driven growth on Truncated), next-gen→legacy generation fallback with
  both reasons named, positive+negative cache keyed (stage, addr, 16 head
  bytes) so a failing shader warns ONCE; XPS5X_DUMP_SHADERS forensic dumps
  (work even when translation fails). OffscreenDrawSink: untranslatable
  shader = skipped draw (counted, debug-logged), DCB continues; embedded
  fixture path intact (M2 gate untouched). Acceptance:
  tests/shader_memory_phase2.rs — DCB binds a real guest-memory PS via SH
  registers → CP → fetch → recompile → Vulkan draw → green pixel readback +
  frame PPM; garbage bind skips the draw, DCB survives.
  Also fixed: guest_mem read used copy_nonoverlapping; a wild-but-committed
  guest range can overlap the destination Vec (page-granular validation) —
  intermittent STATUS_STACK_BUFFER_OVERRUN under test; now ptr::copy.
  Title measurement (PPSA17221, 3×120 s runs, XPS5X_DUMP_SHADERS+FRAMES set):
  **0 shaders fetched, 0 draws — title dies ~10 s in, pre-graphics**, on the
  known std::out_of_range phase-1-unwinding wall above (first failing HLE
  call sceKernelGetdents → 0x8002000e). The GPU-side path is armed and proven;
  re-measure the moment the boot wall falls.

- ASTRO.BOT scene-shader opcode batch (2026-07-18, commit pending; 258/258
  kyty-graphics, 129/129 xps5x-gpu, 276/276 xps5x-hle tests; 1 diagnostic GPU
  test ignored; kyty-graphics+xps5x-gpu clippy clean; GUI build green):
  closed seven title-measured translation blockers with typed decode + SPIR-V
  acceptance coverage. `S_GETPC_B64` now materializes the absolute address of
  the following instruction (including guest bases above 4 GiB);
  `S_PACK_LL_B32_B16` packs the two low halfwords; VOP1 SDWA OMOD preserves
  x2/x4/x0.5 multipliers; address-only `BUFFER_LOAD_DWORDX4` uses zero index;
  `IMAGE_GET_RESINFO dmask=xy` emits `OpImageQuerySizeLod`; `DS_APPEND` and
  `DS_CONSUME` accept their unused non-zero addr field; and
  `IMAGE_SAMPLE_C_LZ` consumes Gen5 `{reference,x,y}`, samples at LOD zero,
  performs the manual `reference <= red` comparison used by SharpEmu, then
  applies all seven supported dmask layouts. The comparison module assembles
  and validates through Naga. Kyty remains behind the xps5x-gpu contract.
  Fresh ASTRO.BOT frame measurement is not yet attributable to this batch:
  two provider-specific ABI aliases (`libSceLibcInternal` C ABI and libkernel
  POSIX pthread names) fixed earlier link stops, but the current architecture
  branch then spins after module_start in native libc allocator mutex traffic
  (443,720 balanced HLE calls/30 s, last HLE scePthreadMutexUnlock), before AGC
  submission. Artifact: `scratch/astro-opcodes-20260718-201535/`; fix that boot
  regression before claiming a translated-shader or frame-count improvement.

- ASTRO.BOT frame path — texture format 71 + validation-layer perf cliff
  (2026-07-19, commit pending; xps5x-gpu 133/133 green, clippy clean).
  DO NOT RE-DERIVE the following; all measured against the retail title with
  `XPS5X_SKIP_MAIN_INIT=1 XPS5X_RESUME_ON_MISSING=1`:
  * The title's GPU work is DETERMINISTIC across runs and across commits
    21483ef/e77ea3f/HEAD: 256 DCB submissions, 1028 draws, 1497 dispatches,
    36 VideoOut flips. "captured AGC submission" is rate-limited (logs the
    first few + powers of two) — counting those lines UNDERCOUNTS submissions
    by ~85x. Read "AGC submission progress" for real totals.
  * The 21483ef-vs-HEAD "frame dump regression" was NOT a dump-path or
    submission-routing bug. `ctx.gpu` (e77ea3f moved submits off
    `AgcGpuSession::global()`) is correctly wired: runtime/thread.rs:410
    calls `new_process` then `install_process`, so global() is the same
    session. Queue routing is also correct (only GpuQueue::AsyncCompute maps
    to is_compute).
  * Real cause: e77ea3f's buffer/texture reclassification correctly began
    treating a T# as a texture, exposing unified format 71 as unimplemented.
    Every draw in the 7966-dword DCB then failed at DWORD 5053 (56 skips).
    FIXED: unified 71 -> (dataFormat 12 = 16_16_16_16, numFormat 7 = FLOAT)
    = R16G16B16A16_SFLOAT @ 8 bpp, verified against SharpEmu
    Gfx10UnifiedFormat.cs:83 and its Gen5 layout table
    (Gen5SpirvTranslator.cs:2667 "SetLayout(12,0,0,16) // 16_16_16_16").
    8 bpp is bpp_log2 3, a table row no 1/4-byte format had exercised;
    tile mode 27 (SW_64KB_R_X) already supported it. Draw skips 56 -> 0.
  * `agc_exec.rs` hard-coded `VulkanBackend::new(true)`, so the Khronos
    validation layer was ALWAYS on for titles: ~0.9 s per
    vkCreateGraphicsPipelines, i.e. ~15 min for 1028 draws — the reason a
    260 s run reached only one submission and looked like a GPU-worker hang.
    Now opt-in via `XPS5X_VULKAN_VALIDATION=1`.
  * STILL NO FRAME, and the remaining wall is now unambiguous: pixel/compute
    shader translation. 16 shaders translate OK, 16 fail, and a draw needs
    its PS, so every draw skips and `sink.last` stays None. Both failure
    reasons share ONE root cause — Extended User Data:
    `ps: user_sgpr > user sgpr count` (analysis.rs:2188 — the shader declares
    more user SGPRs than were ever written; `count` is a high-water mark,
    hw_regs.rs:714) and `ShaderUserData eud_size_dw != 0`. This is backlog
    task #9 (EUD/SRT scalar resolver), NOT a dump/present/format problem.
  * The two frames 21483ef produced were the known static green loading
    composite, never the 3D scene — recovering that dump is not a scene frame.
  Pre-existing (not from this work): `cargo fmt --all --check` fails on
  committed `crates/kyty-graphics/src/run.rs:1262`; belongs to task #12.

## 2026-07-20 — Minecraft (PPSA17221) deep-boot session: NID surface 161→138, AGC frames flowing, audio unblocked

Context: first full `--run-eboot` measurement of Minecraft PS5 (254 MB eboot
+ 5 bundled .prx). Boot reached ~30 threads (MINECRAFT MAIN THREAD, cohtml,
FMOD, RakNet, Agc*) but died twice deterministically: (1) a net thread
jumped to the unresolved-import stub for `socket` from libScePosix;
(2) AgcSubmissionThread deref'd 0x25 after `sceKernelWaitEqueue` timed out.
Uncommitted (user hasn't asked for a commit).

DONE (xps5x-hle, 285/285 tests, workspace clippy clean):
- libScePosix provider aliases — provider-aware resolution meant the
  libkernel-only registrations didn't satisfy libScePosix imports:
  socket/bind/connect/getsockname/inet_pton (kernel_socket.rs) and
  open/read/write/close/lseek (libkernel.rs, same pattern as the existing
  libScePosix::getdents alias). Fixes the `socket` jump-to-stub fault.
- libSceNet socket family (libsce_net.rs): sceNetSocket/SocketClose/Bind/
  Connect/Listen/Accept/Send/Recv/Shutdown/Setsockopt/ErrnoLoc/
  ResolverStartNtoa — offline semantics matching kernel_socket (connect →
  ENETUNREACH, accept/recv → EWOULDBLOCK, send discards validated payload);
  error codes from shadPS4 net_error.h (0x8041_01xx); ErrnoLoc shares the
  __error cell.
- `__stack_chk_fail` registered under libkernel provider (libc.rs).
- libSceAudioOut2 streaming trio (libsce_audio_out2.rs):
  sceAudioOut2ContextPush/ContextAdvance pace to one hardware grain
  (grain_samples/frequency) via a per-context PaceAdvance ported from
  SharpEmu's ContextState (FMOD uses Push as its submission clock);
  sceAudioOut2PortSetAttributes accepted inert; plus
  sceAudioOut2ContextGetQueueLevel (dlsym-only import, never in the static
  table — level always 0 since pacing is synchronous).
  Result: missing NIDs 161 → 138 for this title.
DONE (xps5x-gpu, 121/121 tests):
- vulkan/instance.rs: enable + require vertexPipelineStoresAndAtomics —
  Gen5 vs writes storage buffers; pipeline creation was failing validation
  (VUID-RuntimeSpirv-NonWritable-06341) for whole draw batches.
- draw_translate.rs: Gen5 unified vertex format 11 → R16_UINT (SharpEmu
  Gfx10UnifiedFormat: 11 → (2,4)); the per-frame DWORD 950/1270 skips are
  gone, total_draws 1351 → 2244 at submissions=128.

MEASURED STATE NOW: the title boots deep and runs for minutes — RakNet up
("IPv4 supported"), FMOD mixer paced, cohtml UI active, DCB+ACB submissions
continuous (draws 2244, dispatches 2123, flips 128 at ~1.5 min), ~6/7
shaders translate (16k-word SPIR-V), frames present to scanout. STILL BLACK
frames (1920x1080 flips land but all-zero) — visible output blocked by:
- `libSceAgc` NID 0xfca47359e915d76d (-KRzWekV120) UNKNOWN fn, 14 args,
  stub returns 0 — in NO name DB (ours/SharpEmu/Kyty); likely
  flip/display-chain related. Needs RE from the call site.
- vs translate fail: Recompile_Fetch registers_num=0 (attrib 3); cs:
  ShaderUserData eud_size_dw != 0 (same EUD gap as backlog task #9).
- VGT_PRIMITIVE_TYPE 0 unsupported (single early draw; NONE = no-op on HW).
- sceAcmContextCreate / sceAjmBatch* UNKNOWN ABI (audio decode → silence,
  non-fatal).
- Mass "unknown context register — write skipped" (kyty_graphics::run).
NEXT: identify NID 0xfca47359e915d76d; EUD/SRT scalar resolver (task #9);
prim-type NONE as no-op; dump later flip indices (dump only catches ≤8/2^n
and the flip counter stops ~6-8 presents).

- ASTRO.BOT draw path — Gen5 32 user-SGPRs + two texture gaps (2026-07-19,
  commit pending; workspace 1430/1430 green, clippy clean on touched crates).
  The DRAW path went from "every draw fails" to "draw_skips=0, draw shaders
  translate, 17 shaders translated_ok / 2954 cache hits". Chain, each step
  measured against the retail title then fixed with a red-green test:
  1. **texture format 71** -> R16G16B16A16_SFLOAT @ 8bpp (SharpEmu
     Gfx10UnifiedFormat.cs:83 = dataFormat 12 (16_16_16_16) + numFormat 7
     (FLOAT)). Was failing all 56 draws at DWORD 5053.
  2. **Vulkan validation was hard-coded ON** for titles (agc_exec.rs
     `VulkanBackend::new(true)`): ~0.9s per vkCreateGraphicsPipelines, ~15 min
     for 1028 draws — it LOOKED like a hung GPU worker. Now opt-in via
     `XPS5X_VULKAN_VALIDATION=1`. Check this first if the worker seems stuck.
  3. **Gen5 graphics stages have 32 user SGPRs, not 16.** `UserSgprInfo::
     SGPRS_MAX` was 16, so `set()` silently DROPPED slots 16..31 and capped
     `count` at 16, while ASTRO.BOT pixel shaders declare rsrc2.user_sgpr =
     20/24/30/32 -> `shader_parse_ps` rejected every one. Widened SGPRS_MAX and
     run.rs `set_shader_register`'s SGPRS to 32. No SH-register collision: PS
     user data 0x0C with next SH reg VS_0 at 0x4C, GS_0 0x8C to ES 0xC8.
     Compute stays 16 (COMPUTE_USER_DATA_0..15) and spills to EUD.
     **PS user_sgpr failures 20 -> 0.**
  4. **sharp start_register bound**: Kyty treats >=16 as "lives in the extended
     buffer", but measured PS shaders reference a sharp at start_register=16
     while declaring NO extended buffer — with no EUD the data can only be a
     direct register, so the bound is the register-file size. Self-validating:
     an unwritten slot keeps type_ Unknown and is still rejected.
  5. **texture format 22** -> R32_SFLOAT @ 4bpp (Gfx10UnifiedFormat.cs:48 =
     dataFormat 4 (32) + numFormat 7), a 1920x1080 linear-depth target.
  6. **tile mode 24 = SW_64KB_Z_X** (the DEPTH swizzle; interleaves X/Y from
     bit 0 where _R_X runs X first). Transcribed `RB_PLUS_64K_DEPTH_X` from
     SharpEmu GnmTiling.cs `RbPlus64KDepthX` (GPL-2.0, same source/topology as
     the existing two tables); round-trip test also asserts it is a DIFFERENT
     permutation from mode 27 so a copy-paste slip would fail.
  STILL NO FRAME. State after the chain: no draw failures and no draw skips
  remain, but `execute_dcb_cp` still returns Ok(None) — `sink.last` is never
  set, so no RenderedImage reaches present/dump. NOT yet isolated; a 70 s
  debug run reached only 128 submissions and never got to a draw-carrying
  one, so its "0 draw_common calls" is INCONCLUSIVE, not evidence. Next step
  is a long run under `RUST_LOG=warn,xps5x_gpu=debug` to see whether
  `draw_common` is entered and, if so, which of its early returns fires
  (prime suspect: `color_output_disabled(ctx)` -> "draw consumed without
  colour output (depth path pending)", draw_translate.rs:1080).
  Remaining shader failures are COMPUTE only: EUD (`eud_size_dw != 0`) plus
  opcodes `s_not_b64`, `s_brev_b32`, `v_cmpx_eq_f32`.

- ASTRO.BOT frame dump — the gate was counting the WRONG THING (2026-07-19,
  commit pending; workspace 1430/1430, clippy 0 warnings).
  `maybe_dump_frame` sampled on "draw_index <= 8 || is_power_of_two", but the
  value passed was the CUMULATIVE draw count, which a submission advances by
  its whole draw total at once (`*draws += drawn`): 0 -> 21 -> 42 -> 56, so it
  almost never landed on the cadence. **56 real Vulkan draws rendered and every
  dump was skipped.** Fixed with a `PRESENT_INDEX` incremented once per
  presented frame. Result: 0 -> **10 frames** dumped (1920x1080 and 2432x1368).
  MEASUREMENT TRAP that cost a full diagnostic cycle: the app filters logs with
  **`XPS5X_LOG`, not `RUST_LOG`** (xps5x-core/src/logging.rs:160). Two debug
  runs under RUST_LOG produced zero `debug!` output, which read exactly like
  "draw_common is never entered" — it was an artifact. Always use XPS5X_LOG.
  With it: 56 "drove a register-state Vulkan draw", 57 depth-only draws
  consumed, 17 shaders translated_ok, 2954 cache hits.
  HONEST RESULT: the dumped frames are a SINGLE FLAT COLOUR across the whole
  2432x1368 surface (sampled at 5/20/40/60/80/95% — distinct_pixels=1 at every
  point; consecutive frames byte-identical). This is the previously-known
  loading composite, NOT the 3D scene, exactly as
  [[astro-bot-boot-state]] predicted: the scene is COMPUTE-rendered and the
  compute path still fails on EUD (`eud_size_dw != 0`) plus opcodes
  `s_not_b64`, `s_brev_b32`, `v_cmpx_eq_f32`. Draws are no longer the blocker;
  compute is. Do NOT claim a scene frame from these dumps.

- ASTRO.BOT compute opcodes — s_not_b64 / s_brev_b32 / v_cmpx_eq_f32
  (2026-07-19, commit pending; kyty-graphics 260/260, workspace 1431/1431,
  clippy 0). All three were blocking compute-shader translation. Wired through
  the full chain the codebase requires: types.rs variant -> parse.rs arm(s) ->
  recompile.rs function + dispatch row -> table count assertions bumped
  (245->248 rows, 233->236 implemented) -> per-opcode test.
  * `s_not_b64` (SOP1 0x08, Sdst2Ssrc02): D.u64 = ~S0.u64, SCC = (D != 0).
    New `recompile_snot_b64` modeled on `recompile_swqm_b64` (same format,
    same exec-passthrough/paired-dword/execz/scc structure), OpNot per dword.
  * `s_brev_b32` (SOP1 0x0b, SVdstSVsrc0): OpBitReverse, and it does NOT write
    SCC — pinned by `S::None` in the row and asserted in the SCC-semantics test.
  * `v_cmpx_eq_f32` (VOPC 0x12): the cmpx block mirrors cmp at +0x10, so 0x12
    is the exec-writing eq. Reuses the generic `recompile_vcmpx_xxx_f32` with
    `p1("OpFOrdEqual")` (matching VCmpEqF32's op). Wired at BOTH VOPC parse
    sites (parse.rs:1061 and :1708) — there are two.
  Test uses the MEASURED encoding from the title's failure log
  (`raw 0xbefe087e` = `s_not_b64 exec, exec`).
  MEASURED RESULT: **zero unknown-opcode failures remain** in an ASTRO.BOT run
  (previously 3 distinct). Compute now fails on exactly two things:
  `ShaderUserData eud_size_dw != 0` (8) and `read-only texture type 8 is not
  Texture2D` (type 8 = 1D). Frames still dump (10) and are still a FLAT COLOUR
  (distinct_pixels=1 at 10/35/60/85%) — EUD still gates the scene compute, so
  this did NOT by itself produce scene pixels. **EUD (task #9) is now the
  single remaining blocker on the compute path.**
  Note: `cargo fmt --all --check` flags `crates/xps5x-hle/src/libsce_audio_out2.rs`
  (2 spots) — NOT from this work; it arrived via commit 3d183a1.

- EUD increment 3 — OPEN QUESTION found while scoping (2026-07-19, no code
  change). Do not start increment 3 assuming the existing indexing is right:
  measured `cs@0x50068ef00` has `eud=8` dwords but `sharp[1.0]` reports
  `offset_dw=32`. `read_sharp_fields`'s extended branch reads
  `ext[start - 16 + j]` = `ext[16]`, which is OUT OF RANGE for an 8-dword EUD
  ("extended (EUD) buffer too small"). So either the `-16` rebase is wrong for
  Gen5, or `offset_dw` is not a dword index into the EUD, or the EUD is larger
  than `eud_size_dw` suggests. SETTLE THIS FIRST — wiring a buffer under the
  current assumption would silently feed wrong descriptors (no error, wrong
  pixels). Also note SharpEmu does NOT use a fixed EUD-base convention: it runs
  a full scalar evaluator (`Gen5ShaderScalarEvaluator`, ~2000 lines) to resolve
  descriptors, so there is no short convention to transcribe; our
  `find_scalar_load_bases` + `scalar_load_target_address` (increments 1-2) are
  the right road. `shader_get_input_info_vs` has `mem` but NOT a parsed
  `ShaderCode`, so increment 3 needs a parse pass threaded in.

- EUD increment 3 — PARTIALLY LANDED (2026-07-19; kyty-graphics 260/260,
  workspace 1431/1431, clippy 0). Two changes, both evidence-driven:
  1. **The EUD rebase is the user-SGPR FILE SIZE, not a literal 16.** Kyty's 16
     was the PS4 user-SGPR count. Measured: a shader with eud_size_dw=8 places
     a sharp at offset_dw=32 while its other sharps sit direct at s0/s8 — under
     `-16` that reads ext[16] (past the end of an 8-dword EUD); under
     `-32` (= SGPRS_MAX) it reads ext[0], the first descriptor, which fits.
     `read_sharp_fields` now rebases by `UserSgprInfo::SGPRS_MAX`. This RESOLVES
     the open question logged in the previous entry.
  2. New `read_extended_user_data` reads `eud_size_dw` dwords from the sgpr pair
     immediately AFTER the declared user SGPRs (measured: declared=14, s14:s15
     -> descriptor-shaped data). `shader_parse_usage2` takes the buffer as a new
     `eud` parameter and feeds the extended branch that already existed.
     Refuses (does not invent descriptors) when the pointer is null/unaligned/
     unmapped or the pair is out of range.
  MEASURED: EUD failures 8 -> 6, shader translate_failed 15 -> 11. So the
  "pair right after declared user SGPRs" convention is CORRECT FOR SOME SHADERS
  BUT NOT ALL — the remaining 6 report "(EUD unreadable)". Those need the base
  recovered by scalar-load analysis (`find_scalar_load_bases` +
  `scalar_load_target_address`, increments 1-2, already built): parse the
  shader, find the `s_load_dwordx*` whose base is a user-SGPR pair, and use
  that address. `shader_get_input_info_vs` has `mem` but no parsed
  `ShaderCode`, so that step needs a parse pass threaded in.
  STILL NO SCENE: 10 frames, still flat colour. Scene compute needs the
  remaining 6 EUD shaders plus texture type 8 (1D).

## 2026-07-20 (cont.) — black-frame root-cause chain: draws execute, pixels never land

Follow-ups after the NID/audio/Vulkan batch (all uncommitted; a concurrent
session landed complementary Gen5 work mid-flight: 32-slot user SGPRs,
PRESENT_INDEX dump cadence, tiling, ASTRO formats):
- Unknown libSceAgc NID 0xfca47359e915d76d = SharpEmu's
  `sceAgcDriverUnknown_KRzWekV120` (AgcExports.cs:2821) — trace-and-return-OK
  stub there too; our return-0 matches reference behavior. NOT the
  black-frame cause.
- Gen5 vertex format 57 → R8G8B8A8_SNORM (draw_translate.rs; SharpEmu table
  57 → (10,1)). Removed the per-frame DWORD-950 skips; draws 2244 → 10142+
  per run, zero draw-skip warnings in later runs.
- DECISIVE census (XPS5X_DUMP_ALL_TARGETS): at presents 5..128 every render
  target (0x1f7d0000, 0x31c10000, 0x20040000) reports non_black_pixels = 0.
  Black frames are NOT a flip/scanout mismatch — the draw pipeline produces
  zero pixels, including no clear-alpha.
- XPS5X_DUMP_GPU_RESOURCES: vertex buffer @0x313f0150 (stride 28) holds REAL
  plausible UI data (0/1/2/3 floats) — vertex fetch works.
- The bound UI texture @0x31c00000 (1920x1080, fmt 56, tile 27) reads back
  100% zeros. NOT a PM4 DMA problem: captured DCB layouts carry opcodes
  16/70/118/38/19/53/46/44/7 — no IT_DMA_DATA (0x50) or IT_WRITE_DATA (0x37)
  at all, so the texture is meant to be filled by draws/compute, i.e. the
  same zero-output root cause. (Note: IT_DMA_DATA/IT_WRITE_DATA ARE
  consumed-without-effect in kyty_graphics::run — a real gap for OTHER
  titles, just not this one's current path.)
- Localization: rasterization executes but nothing lands. Vertex data OK,
  shaders translate, pipelines create, targets exist. Remaining suspects in
  order: (1) shader data path — SRT/EUD/user-SGPR delivery to vs/ps
  (cs@0x253a5000 still fails on ShaderUserData eud_size_dw != 0; concurrent
  session is mid-fix on the 32-slot SGPR widening), (2) skipped context
  regs incl. CB_SHADER_MASK 0x8f and CB_COLOR slot regs 0x366-0x3af,
  (3) vs Recompile_Fetch registers_num=0 (attrib 3) — 19 skips.
NEXT: finish EUD/SRT scalar resolution (task #9, concurrent session),
then re-measure census; VGT_PRIMITIVE_TYPE 0 → clean no-op (currently a
warn-and-skip, semantically right); consider honoring CB_SHADER_MASK.

- EUD strategy 2 (scalar-load base) — LANDED BUT INEFFECTIVE, measured
  (2026-07-20; kyty-graphics 260/260, workspace 1431/1431, clippy 0).
  `read_extended_user_data` now falls back to parsing the shader and trying
  `find_scalar_load_bases` + `scalar_load_target_address` when the
  after-declared pair fails. **MEASURED: no change — still 6 "EUD unreadable".**
  So for those shaders the scalar-load scan yields no usable base. Next
  diagnostic (do this BEFORE writing more resolver code): log, for one failing
  shader (`cs@0x50053c700`, eud=20 declared=14 count=14), whether
  (a) `shader_parse` of its `data_addr` even succeeds here,
  (b) how many `SLoadDword*` `find_scalar_load_bases` returns, and
  (c) what addresses those bases compute to and whether `dwords_at` backs them.
  One of those three is failing and the log will say which; guessing further
  resolver strategies without it is wasted work.
  Evidence recap for the failing class: `count == declared` (no register past
  the declared file holds the pointer) and sharps at offset_dw 32/40/48 with
  eud=20 dwords -> ext[0]/ext[8]/ext[16] under the SGPRS_MAX rebase.
  NOTE the heuristic risk now in the tree: strategy 2 takes the FIRST readable
  scalar-load target. It is currently inert (never succeeds on this title), but
  if a later title starts hitting it, wrong descriptors would surface as wrong
  pixels rather than an error.

- EUD strategy 2 — ROOT CAUSE FOUND, one layer left (2026-07-20;
  kyty-graphics 260/260, workspace 1431/1431, clippy 0).
  Added a `TRACE_EUD2` diagnostic (gated on XPS5X_TRACE_EUD) that reports, per
  failing shader, whether the code is mapped, whether `shader_parse` succeeds,
  how many `SLoadDword*` were found, and whether each computed base address is
  backed. It answered immediately: **`shader_parse` was FAILING**, so
  `find_scalar_load_bases` returned nothing — strategy 2 was never actually
  running. Two successive causes, first now FIXED:
  1. FIXED: `unknown sopp opcode 0x1f (raw 0xbf9f0000)` = RDNA2 `s_code_end`,
     the padding terminator emitted AFTER real shader code. Now parsed as an
     end-of-code marker (parse.rs SOPP arm).
  2. OPEN: with s_code_end decoding, the parse CONTINUES past it into padding
     and dies on `unknown operand: 115` (= ttmp7, i.e. garbage decoded as an
     instruction). So `shader_parse` does NOT stop at a terminator when handed a
     whole fetched buffer. **NEXT STEP: give the full-buffer parse a hard stop
     at SEndpgm/s_code_end** (or have `read_extended_user_data` pre-truncate
     `src` at the first terminator before calling shader_parse). That is the
     only thing between here and strategy 2 actually executing — it has never
     yet run on this title, so its effectiveness is still UNMEASURED.
  Frames still 10, still flat colour; the 6 EUD shaders and texture type 8 (1D)
  remain the compute blockers.

- EUD strategy 2 — EXACT REMAINING BLOCKER IDENTIFIED (2026-07-20;
  kyty-graphics 260/260, workspace 1431/1431, clippy 0).
  `shader_parse` DOES NOT STOP AT A TERMINATOR. Proof: pre-truncating the
  fetched window at s_endpgm/s_code_end and keeping the terminator gave
  "truncated instruction at 0x2e0"; keeping ONE MORE dword moved the error to
  0x2e4 — exactly one dword later. It parses until the buffer is exhausted and
  always fails at the tail, so NO truncation length can satisfy it.
  **THE FIX IS INSIDE `shader_parse`: end the instruction loop after SEndpgm**
  (s_code_end now maps to SEndpgm, parse.rs SOPP 0x1f). Deliberately NOT done
  here: that loop is the main shader path for every title, and changing it
  needs a full re-measure of Minecraft + ASTRO.BOT, not a blind edit.
  Until then strategy 2 (scalar-load EUD base recovery) CANNOT run — it has
  still never executed on this title, so its effectiveness remains UNMEASURED
  and the 6 "EUD unreadable" shaders are NOT evidence against it.
  Landed and safe meanwhile: s_code_end parsing, the SGPRS_MAX EUD rebase, the
  after-declared-pair EUD read (fixed 2 of 8 shaders), and the TRACE_EUD2
  diagnostic that produced all of the above.

- **EUD RESOLVED (task #9 closed)** (2026-07-20; kyty-graphics 260/260,
  workspace 1432/1432, clippy 0). `ShaderUserData eud_size_dw != 0` no longer
  appears in an ASTRO.BOT run — it was the blocker for 8 shaders.
  Root cause was NOT the resolver logic but `shader_parse`'s end detection.
  Kyty breaks at s_endpgm (0xBF81_0000) *unless a live label targets past it*
  — correct for multi-block shaders, which ASTRO.BOT's are. Those kept parsing
  (rightly) until they reached RDNA2 `s_code_end` (0xBF9F_0000), which had no
  break, so the parse ran on into padding and died ("unknown operand: 115").
  FIX: `s_code_end` ends the code BLOCK and takes NO live-label exception —
  nothing can branch past it — so it is now an unconditional break in the same
  end-detection condition (parse.rs), plus a SOPP 0x1f arm mapping it to
  SEndpgm. Truncating the buffer in the caller does NOT work and was reverted:
  the loop parses to exhaustion, so trimming just moves the tail error one
  dword (proven: 0x2e0 -> 0x2e4).
  With the parse fixed, EUD strategy 2 RUNS for the first time and succeeds:
  cs@0x50053c700 need=20dw finds 8 scalar loads (all backed), ps@0x500652400
  need=12dw finds 2. Combined with strategy 1 (pair after declared user SGPRs,
  for shaders where count > declared) all EUD shaders now resolve.
  **STILL NO SCENE**: 10 frames, still flat colour (distinct_pixels=1 at
  10/30/50/70/90%). translate_failed is still 11 and 2 draws want texture
  type 8 (1D). So EUD was necessary but NOT sufficient — the next measurement
  should re-enumerate what those 11 failures actually are now, since the whole
  failure profile shifted when the parser stopped truncating shaders early.

- Post-EUD failure re-enumeration (2026-07-20) — the remaining shader
  translation failures are FIVE DISTINCT FEATURES, not one gap. Measured after
  the s_code_end parse fix (which changed the profile, so pre-fix lists are
  stale — this supersedes them):
    2x `mubuf feature: idxen == 0`        (vs@0x50074e000, cs@0x500757800)
    2x `ds feature: addr != 0`            (cs@0x5006c5f00, cs@0x5005fd000)
    2x `unknown usage type: 0x05`         (ps@0x500652400, cs@0x500690400)
    1x `unknown exp target: 0x0d`         (vs@0x100008e6cd00)
    1x `read-only texture type 8` (1D)    (ps@0x500640200)
  Each is an independent recompiler/analysis add. **No single remaining fix
  produces the scene** — plan for five, then re-measure. Note the DS (LDS) and
  MUBUF-idxen items are compute-side, i.e. on the path the scene actually
  renders through.

- Texture type 8 (1D) supported (2026-07-20; xps5x-gpu green). A 1D image is a
  2D image one row tall and the T# already reports height 1, so the existing 2D
  decode path handles it unchanged (measured: a 1x1 format-71 tile-27 texture).
  Kept as its own match arm so a >1-row "1D" texture would still be visible.
  Four of the five post-EUD features remain: `mubuf idxen == 0` (vs+cs),
  `ds addr != 0` (cs, LDS), `unknown usage type 0x05` (ps+cs),
  `unknown exp target 0x0d` (vs).
  STILL FLAT: 10 frames, uniform 0x408000, frames byte-identical.
  MEASUREMENT WARNING for the next session: sampling a PPM with
  `od | paste - - -` at an arbitrary byte offset can report a spurious extra
  "distinct pixel" from a truncated trailing group at the sample boundary. It
  briefly looked like content had appeared (distinct=2) — it had not. Verify
  any apparent content with a full-histogram check, not a boundary sample.

- Sizing the remaining four (2026-07-20, inspection only, no code change).
  Two of the four are SUBSYSTEMS, not opcode adds — do not schedule them as
  quick wins:
  * `ds addr != 0` — the DS parser (parse.rs ~2717) supports ONLY
    DS_APPEND/DS_CONSUME (GDS counters via M0); every real LDS access is gated
    off (addr/data0/data1/offset0/offset1 must all be 0, gds must be 1). So
    this is not a flag to relax: it means **LDS/shared memory is entirely
    unimplemented** — needs SPIR-V Workgroup storage, LDS size from
    COMPUTE_PGM_RSRC2.lds_size, and barrier handling. ASTRO.BOT's scene compute
    uses it.
  * `mubuf idxen == 0` — an ADDRESSING-MODE change (address becomes
    soffset+offset, plus a VGPR when offen), touching how every buffer load
    resolves for every title. Needs a Minecraft + ASTRO.BOT re-measure to prove
    it does not corrupt the currently-working buffer path.
  The other two (`unknown usage type 0x05`, `unknown exp target 0x0d`) are
  unsized.
  HONEST FORECAST: clearing all four is multi-session, and even then ASTRO.BOT's
  scene additionally depends on the compute -> storage-image writeback front
  (front #3 in [[astro-bot-boot-state]]), which is still unimplemented. Do not
  promise a scene frame from the four alone.

- `unknown exp target 0x0d` sized (2026-07-20, inspection only): EXP target
  0x0c = POS0 (handled), **0x0d = POS1** — the second position export, which
  carries clip/cull distances (and in some encodings point size / viewport /
  layer). Not a parse-table gap: honouring it needs SPIR-V ClipDistance/
  CullDistance declarations. A no-op arm WOULD let the shader translate, but
  silently disables clipping — decide that deliberately, and note it in the
  frame, rather than slipping it in as a "parser fix".
  Remaining four, sized: LDS (subsystem), mubuf idxen==0 (addressing mode,
  needs 2-title re-measure), exp POS1 (clip/cull), usage type 0x05 (unsized).

- `unknown usage type 0x05` sized (2026-07-20, inspection only): the usage-type
  dispatch in `shader_parse_usage2` (analysis.rs ~1070) handles 8 (vertex
  buffer) and 10 (vertex attrib) explicitly, with direct/sharp entries handled
  earlier; type 5 is not in the table and is NOT documented in-tree. Identify it
  from the Gen5 ShaderUserData usage enum (SharpEmu Gen5ShaderIr.cs /
  AgcExports, or Kyty Shader.cpp L1490/L1564 region) BEFORE writing an arm —
  a wrong guess here mis-binds a resource silently.
  All four remaining items are now sized; none is a quick win. See the previous
  three entries.
