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
